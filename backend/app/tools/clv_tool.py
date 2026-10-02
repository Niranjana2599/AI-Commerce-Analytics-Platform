"""CLV model adapter with explicit input-contract validation."""

from typing import Any
import pandas as pd

from backend.app.services import commerce
from backend.app.tools.rfm_tool import run as rfm_run


def run(args: dict[str, Any]) -> dict[str, Any]:
    rows = args.get("customers")
    if rows is not None:
        if not isinstance(rows, list) or not rows:
            raise ValueError("customers must be a non-empty cohort list")
        model = commerce.load_model("clv")
        expected = list(getattr(model, "feature_names_in_", []))
        frame = pd.DataFrame(rows)
        if not expected or not set(expected).issubset(frame.columns):
            raise ValueError("Saved CLV model feature contract is not present in the cohort; no features were inferred")
        output = frame.copy()
        output["predicted_clv"] = model.predict(frame[expected])
        return {"customers":output.sort_values("predicted_clv",ascending=False).to_dict("records"),"model":"saved CLV artifact","feature_contract":expected}
    customer_id, features = args.get("customer_id"), args.get("features")
    if features:
        result = commerce.predict("clv", features)
        return {"customers":[{"customer_id":customer_id,"predicted_clv":result["prediction"]}],"model":"saved CLV artifact"}
    if not customer_id:
        raise ValueError("Provide customer_id and validated model features for CLV prediction")
    customer = rfm_run({"customer_id": customer_id})["customers"][0]
    model = commerce.load_model("clv")
    expected = list(getattr(model, "feature_names_in_", []))
    if not expected or not set(expected).issubset(customer):
        raise ValueError("Saved CLV model feature contract does not match available RFM fields; provide its documented feature mapping")
    prediction = model.predict(pd.DataFrame([{key: customer[key] for key in expected}]))[0]
    return {"customers":[{"customer_id":str(customer_id),"predicted_clv":prediction.item() if hasattr(prediction,"item") else prediction}],"model":"saved CLV artifact"}
