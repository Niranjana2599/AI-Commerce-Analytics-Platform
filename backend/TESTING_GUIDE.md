# Backend testing guide

## 1. Start the API

From the repository root, install the backend runtime and start Uvicorn:

```powershell
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --reload
```

Open Swagger UI at `http://127.0.0.1:8000/docs`. The OpenAPI document is at
`http://127.0.0.1:8000/openapi.json`.

## 2. Test in Swagger UI

Select an endpoint, choose **Try it out**, enter the example body, then select
**Execute**. Start with `GET /api/v1/health`; it does not require data or models.

Use these request bodies:

```json
// POST /api/v1/predictions/churn
{"features":{"Total_Orders":2,"Total_Spend":350,"Avg_Review_Score":4.2,"Avg_Freight":18}}

// POST /api/v1/predictions/clv
{"features":{"Total_Orders":2,"Total_Spend":350,"Frequency":2,"Monetary":350}}

// POST /api/v1/predictions/delivery-delay
{"order_purchase_date":"2026-07-01","estimated_delivery_date":"2026-07-15","customer_state":"SP"}

// POST /api/v1/sentiment
{"review":"The product arrived quickly and works perfectly."}

// POST /api/v1/forecast/demand
{"product_id":null,"days":7}

// POST /api/v1/chat
{"question":"What are the main customer complaints?","limit":5}
```

For recommendations, use `GET /api/v1/recommendations/{customer_id}?limit=10`.
Copy a valid `customer_unique_id` from the processed master dataset if you want
personalized unseen-product filtering.

## 3. Test in Postman

1. Create an environment variable named `baseUrl` with value `http://127.0.0.1:8000/api/v1`.
2. Create requests with URLs such as `{{baseUrl}}/health` and `{{baseUrl}}/sentiment`.
3. Set `Content-Type: application/json` for POST requests.
4. Paste the corresponding JSON body from the Swagger section above.
5. Assert success in the Tests tab, for example:

```javascript
pm.test("returns success", () => pm.response.to.have.status(200));
pm.test("has a prediction", () => pm.expect(pm.response.json()).to.have.property("prediction"));
```

## 4. Run automated smoke tests

```powershell
python -m pytest backend/tests
```

## 5. Docker verification

Build from the repository root:

```powershell
docker build -f backend/Dockerfile -t ai-commerce-api .
docker run --rm -p 8000:8000 ai-commerce-api
```

Then repeat the `/api/v1/health` Swagger or Postman test. The image copies the
prepared Parquet dataset and model artifacts; keep them available at build time.
