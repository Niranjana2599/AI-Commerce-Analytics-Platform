"""Churn model adapter honoring the saved estimator's explicit feature contract."""

from typing import Any
import pandas as pd

from backend.app.services import commerce
from backend.app.tools.rfm_tool import run as rfm_run


def run(args: dict[str, Any]) -> dict[str, Any]:
    rows = args.get("customers")
    if rows is not None:
        if not isinstance(rows, list) or not rows:
            raise ValueError("customers must be a non-empty cohort list")
        model = commerce.load_model("churn")
        expected = list(getattr(model, "feature_names_in_", []))
        frame = pd.DataFrame(rows)
        if not expected or not set(expected).issubset(frame.columns):
            raise ValueError("Saved churn model feature contract is not present in the cohort; no features were inferred")
        if not hasattr(model, "predict_proba"):
            raise ValueError("Saved churn model does not expose predict_proba")
        probabilities = model.predict_proba(frame[expected])[:, -1]
        output = frame.copy()
        output["churn_probability"] = probabilities
        return {"customers":output.sort_values("churn_probability",ascending=False).head(20).to_dict("records"),"model":"saved churn artifact","feature_contract":expected}
    customer_id = args.get("customer_id")
    features = args.get("features")
    if features:
        result = commerce.predict("churn", features)
        return {"customers":[{"customer_id": customer_id,"prediction": result["prediction"], "churn_probability": result["probability"]}],"model":"saved churn artifact"}
    if not customer_id:
        raise ValueError("Provide customer_id and validated model features for churn prediction")
    customer = rfm_run({"customer_id": customer_id})["customers"][0]
    model = commerce.load_model("churn")
    expected = list(getattr(model, "feature_names_in_", []))
    if not expected or not set(expected).issubset(customer):
        raise ValueError("Saved churn model feature contract does not match available RFM fields; provide its documented feature mapping")
    row = pd.DataFrame([{key: customer[key] for key in expected}])
    probability = float(model.predict_proba(row)[0][-1]) if hasattr(model, "predict_proba") else None
    prediction = model.predict(row)[0]
    return {"customers":[{"customer_id":str(customer_id),"prediction":prediction.item() if hasattr(prediction,"item") else prediction,"churn_probability":probability}],"model":"saved churn artifact"}
