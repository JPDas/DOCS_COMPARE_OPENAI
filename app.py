import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from doc_compare import DocumentsCompare

# load environment variables
load_dotenv()
# title of the streamlit app
st.title(f"""Documents Comparision with Azure OpenAI""")

# default container that houses the document upload field
with st.container():
    # header that is shown on the web UI
    st.header('Single File Upload')
    # the first file upload field, the specific ui element that allows you to upload file 1
    File1 = st.file_uploader('Upload File 1', type=["pdf"], key="doc_1")
    # the second file upload field, the specific ui element that allows you to upload file 2
    File2 = st.file_uploader('Upload File 2', type=["pdf"], key="doc_2")
    submit = st.button("Compare",)

    if submit:
        if File1 and File2 is not None:

            # determine the path to temporarily save the PDF file that was uploaded
            save_folder = "Temp"
            # create a posix path of save_folder and the first file name
            save_path_1 = Path(save_folder, File1.name)
            # create a posix path of save_folder and the second file name
            save_path_2 = Path(save_folder, File2.name)
            # write the first uploaded PDF to the save_folder you specified
            with open(save_path_1, mode='wb') as w:
                w.write(File1.getvalue())
            # write the second uploaded PDF to the save_folder you specified
            with open(save_path_2, mode='wb') as w:
                w.write(File2.getvalue())
            # once the save path exists for both documents you are trying to compare...
            if save_path_1.exists() and save_path_2.exists():
                # write a success message saying the first file has been successfully saved
                st.success(f'File {File1.name} is successfully saved!')
                # write a success message saying the second file has been successfully saved
                st.success(f'File {File2.name} is successfully saved!')
                # running the document comparison task, and outputting the results to the front end
                doc_compare = DocumentsCompare(save_path_1, save_path_2)
                doc_1_text, doc_2_text = doc_compare.preprocess()

                prompt = doc_compare.build_prompt(doc_1_text, doc_2_text)
                question_with_prompt = doc_compare.prompt_finder(prompt)

                response = doc_compare.llm_compare(question_with_prompt)
                st.write(response)
                #st.write(doc_compare(save_path_1, save_path_2))
                # removing the first PDF that was temporarily saved to perform the comparison task
                os.remove(save_path_1)
                # removing the second PDF that was temporarily saved to perform the comparison task
                os.remove(save_path_2)
            print("Calling LLM functionality")



