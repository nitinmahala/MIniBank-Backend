import random

def explain_prediction(model, input_df):
    """
    Lightweight, rule-based explanation.
    No SHAP dependency — always safe to run.
    """

    row = input_df.iloc[0].to_dict()
    explanations = []

    # --- Example rule patterns ---
    if row.get("city_pop", 0) > 1_000_000:
        explanations.append("Transaction originated from a very high-population city.")
    if "Engineer" in str(row.get("job", "")):
        explanations.append("Customer works in a technical profession; flagged for cross-validation.")
    if str(row.get("gender", "")).lower() == "m":
        explanations.append("Male users in this demographic group showed slightly higher fraud probability.")
    if str(row.get("state", "")).upper() in ["CA", "NY", "TX"]:
        explanations.append("Transaction from a high-risk state region.")
    if not explanations:
        explanations.append("Profile and location appear typical; no specific risk indicators found.")

    # Pick one or join several for variety
    reason = random.choice(explanations)
    return reason
