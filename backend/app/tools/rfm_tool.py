"""Customer RFM summaries derived by the existing feature function."""

from typing import Any

from backend.app.services.commerce import master_data
from src.features.customer import build_customer_features


def run(args: dict[str, Any]) -> dict[str, Any]:
    customers = build_customer_features(master_data())
    customer_id = args.get("customer_id")
    customer_ids = args.get("customer_ids")
    if customer_ids is not None:
        if not isinstance(customer_ids,list) or not customer_ids or len(customer_ids)>100 or any(not isinstance(value,str) for value in customer_ids):
            raise ValueError("customer_ids must be a list of 1 to 100 strings")
        customers=customers[customers["customer_unique_id"].astype(str).isin(customer_ids)]
        if customers.empty: raise ValueError("No cohort identifiers matched customer features")
        return {"customers":customers.to_dict("records"),"count":int(len(customers)),"feature_columns":list(customers.columns)}
    if customer_id:
        customers = customers[customers["customer_unique_id"].astype(str) == str(customer_id)]
        if customers.empty:
            raise ValueError("Unknown customer_id")
        return {"customers": customers.head(1).to_dict("records"), "count": 1}
    ranked = customers.sort_values(["recency_days", "monetary"], ascending=[False, False]).head(20)
    return {"customers": ranked.to_dict("records"), "count": int(len(customers)), "feature_columns":list(customers.columns), "note": "RFM recency is descriptive, not the trained churn model."}
