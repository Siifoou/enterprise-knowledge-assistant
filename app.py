import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Enterprise Knowledge Assistant", layout="centered")
st.title("Enterprise Knowledge Assistant")

tab_admin, tab_ask = st.tabs(["Admin — Manage Documents", "Ask a Question"])

# ---------------- ADMIN TAB ----------------
with tab_admin:
    st.subheader("Upload a document")

    uploaded_file = st.file_uploader("Choose a PDF or Word file", type=["pdf", "docx"])
    category = st.text_input("Category (optional)", value="")

    if st.button("Upload and index"):
        if uploaded_file is None:
            st.warning("Please choose a file first.")
        else:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            data = {"category": category}
            response = requests.post(f"{API_URL}/documents/upload", files=files, data=data)

            if response.status_code == 200:
                st.success(f"'{uploaded_file.name}' uploaded and indexed successfully.")
            else:
                st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")

    st.divider()
    st.subheader("Indexed documents")

    docs_response = requests.get(f"{API_URL}/documents")
    if docs_response.status_code == 200:
        docs = docs_response.json()
        active_docs = [d for d in docs if d["status"] == "indexed"]
        failed_docs = [d for d in docs if d["status"] == "failed"]

        if not active_docs:
            st.info("No documents indexed yet.")
        else:
            for doc in active_docs:
                col1, col2, col3 = st.columns([3, 2, 1])
                col1.write(doc["name"])
                col2.write(f"{doc['category'] or '—'} · {doc['status']}")
                if col3.button("Delete", key=f"delete_{doc['id']}"):
                    del_response = requests.delete(f"{API_URL}/documents/{doc['id']}")
                    if del_response.status_code == 200:
                        st.success(f"Deleted '{doc['name']}'.")
                        st.rerun()
                    else:
                        st.error("Delete failed.")

        if failed_docs:
            with st.expander(f"⚠️ Failed uploads ({len(failed_docs)})"):
                for doc in failed_docs:
                    col1, col2 = st.columns([3, 1])
                    col1.write(f"{doc['name']} — upload failed (likely scanned/empty)")
                    if col2.button("Remove", key=f"delete_failed_{doc['id']}"):
                        requests.delete(f"{API_URL}/documents/{doc['id']}")
                        st.rerun()

# ---------------- ASK TAB ----------------
with tab_ask:
    st.subheader("Ask a question about the indexed documents")

    question = st.text_input("Your question")

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please type a question first.")
        else:
            with st.spinner("Searching documents and generating answer..."):
                response = requests.post(f"{API_URL}/ask", data={"question": question})

            if response.status_code == 200:
                result = response.json()
                st.markdown(f"**Answer:** {result['answer']}")

                if result["sources"]:
                    st.divider()
                    st.caption("Sources")
                    for src in result["sources"]:
                        with st.expander(src["doc_name"]):
                            st.write(src["excerpt"])
            else:
                st.error("Something went wrong — check that the API server is running.")