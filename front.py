import streamlit as st
import requests


API_URL = "https://charanss1-shl-assessment-recommender.hf.space/recommend"


st.set_page_config(page_title="SHL Assessment Recommender", layout="centered")

st.title("SHL Assessment Recommendation System")
st.write(
    "Enter a job description or hiring requirement below, and the system will "
    "recommend suitable SHL assessments."
)

#
query = st.text_area(
    "Job Description / Query",
    placeholder="Hiring a Java developer with good collaboration skills...",
    height=150
)

if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Fetching recommendations..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"query": query},
                    timeout=30
                )

                if response.status_code != 200:
                    st.error(f"API Error: {response.status_code}")
                else:
                    data = response.json()
                    recommendations = data.get("recommendations", [])

                    if not recommendations:
                        st.info("No recommendations found.")
                    else:
                        st.success("Recommended Assessments:")

                        for i, rec in enumerate(recommendations, start=1):
                            st.markdown(f"### {i}. {rec['assessment_name']}")
                            st.markdown(f"- Test Type: {rec['test_type']}")
                            st.markdown(f"- Assessment Length {rec['assessment_length']}")
                            st.markdown(f"- Remote Testing: {rec['remote_testing']}")
                            st.markdown(f"-Adaptive Support: {rec['adaptive_support']}")
                            st.markdown(f"- URL: {rec['url']}")
                            st.markdown("---")

            except Exception as e:
                st.error(f"Failed to connect : {e}")

