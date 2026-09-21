import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Sydney Housing Price Predictor", page_icon="house")

bundle = joblib.load("housing_models.joblib")
rf, lr = bundle["rf"], bundle["lr"]
columns = bundle["columns"]
house_median_land = bundle["house_median_land"]

st.title("Sydney Housing Price Predictor")
st.caption(
    "Trained on 118 properties sold in Blacktown, Parramatta and Strathfield. "
    "Typical error is about 10% of the sale price."
)

col1, col2 = st.columns(2)

with col1:
    suburb = st.selectbox("Suburb", ["Blacktown", "Parramatta", "Strathfield"])
    type_group = st.selectbox("Property type", ["Apartment", "House & other"])
    bedrooms = st.number_input("Bedrooms", 1, 7, 3)
    bathrooms = st.number_input("Bathrooms", 1, 5, 2)
    parking = st.number_input("Car spaces", 0, 6, 1)

with col2:
    if type_group == "Apartment":
        land_size, land_missing = 0.0, 0
        st.info("Apartments are recorded with a land size of 0 m².")
    else:
        land_unknown = st.checkbox("Land size unknown")
        if land_unknown:
            land_size, land_missing = house_median_land, 1
            st.warning(f"Using the median house land size ({house_median_land:.0f} m²).")
        else:
            land_size = st.number_input("Land size (m²)", 50.0, 2000.0, 550.0, step=10.0)
            land_missing = 0

    granny = st.checkbox("Listing mentions a granny flat")
    pool = st.checkbox("Listing mentions a pool")
    fairwater = st.checkbox("Fairwater estate (Blacktown)")

row = {c: 0 for c in columns}
row.update({
    "bedrooms": bedrooms, "bathrooms": bathrooms, "parking": parking,
    "land_size": land_size, "land_missing": land_missing,
    "mentions_granny_flat": int(granny), "has_pool": int(pool),
    "is_fairwater": int(fairwater),
})
if suburb == "Parramatta":
    row["suburb_Parramatta"] = 1
if suburb == "Strathfield":
    row["suburb_Strathfield"] = 1
if type_group == "House & other":
    row["type_group_House & other"] = 1

X_new = pd.DataFrame([row])[columns]

if st.button("Predict sale price", type="primary"):
    rf_pred = float(np.exp(rf.predict(X_new)[0]))
    lr_pred = float(np.exp(lr.predict(X_new)[0]))
    gap = abs(rf_pred - lr_pred) / min(rf_pred, lr_pred) * 100

    st.subheader("Predicted sale price")
    a, b = st.columns(2)
    a.metric("Random Forest (main model)", f"${rf_pred:,.0f}")
    b.metric("Linear Regression (comparison)", f"${lr_pred:,.0f}")

    if gap > 25:
        st.error(
            f"The two models disagree by {gap:.0f}%. In the error analysis, large "
            "disagreements marked properties the models handle badly, such as blocks "
            "sold for their development potential. Treat this estimate with caution."
        )
    else:
        st.success(f"The two models agree within {gap:.0f}%.")

    if land_missing == 1:
        st.warning(
            "Land size was estimated, not measured. The estimate is one median for all "
            "suburbs, which is usually too low for Strathfield and too high for Blacktown."
        )
    if type_group == "House & other" and suburb == "Blacktown" and land_size > 750:
        st.warning(
            "Large Blacktown blocks are often sold as development sites. The model has no "
            "feature for zoning or building condition, so it can over-value them."
        )

        with st.expander("What this model cannot see"):
        st.write(
            "- Internal floor area, age and condition of the building\n"
            "- Zoning and development potential\n"
            "- Quality of finishes, street prestige, outlook and proximity to amenities\n"
            "- The date of sale: the data spans April 2023 to September 2026 but sale date is not a feature\n"
            "- Investment-only apartments, which were removed before training"
        )
