import streamlit as st
import pandas as pd
import re

def process_html(file_content):
    s = file_content
    
    # extract data of image
    ars_jpg = re.findall(r'<a href="https.*?"', s)
    ars_jpg2 = [re.sub(r'(?<=/s)\d+(?=/)', '4000', s) for s in ars_jpg]
    
    ars_url = re.findall(r'src="(https://[^"]+)"', s)
    do_width = re.findall(r'data-original-width="([^"]+)"', s)
    do_height = re.findall(r'data-original-height="([^"]+)"', s)
    
    # to dataframe
    df = pd.DataFrame(zip(ars_jpg2, do_width, do_height, ars_url), 
                      columns=['ars_url', 'do_width', 'do_height', 'img_url'])
    df[['do_width', 'do_height']] = df[['do_width', 'do_height']].astype(int)
    
    # size_of_image
    df['w'] = df['do_width'].le(df['do_height']).map({True: 'width=auto', False: 'width="350"'})
    df['h'] = df['do_width'].lt(df['do_height']).map({True: 'height="350"', False: 'height=auto'})
    
    # add numbering
    df['seq'] = [f"{i:03}" for i in range(len(df))]
    
    # to_html
    df['all'] = df['ars_url'] + ' target="_blank"><img src="' + df['img_url'] + '" alt="' + df['seq'] + '.jpg" ' + df['w'] + ' ' + df['h'] + '></a>'
    
    return ''.join(df['all'].tolist())

def main():
    st.set_page_config(page_title="HTML Image Converter", layout="wide")
    st.title("📸 HTML Image Extractor & Converter")
    st.write("Convert HTML image data for easy reuse.")
    
    if "output_text" not in st.session_state:
        st.session_state.output_text = ""

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Step 1: Input HTML")
        file_content = st.text_area("Paste From Google Blogger:", height=300)
        process_button = st.button("Convert", help="Click to convert the HTML content")
    
    with col2:
        st.subheader("Step 2: Output HTML")
        if process_button and file_content:
            st.session_state.output_text = process_html(file_content)
        
        output_text = st.session_state.output_text
        st.text_area("For Seesaa Blog:", value=output_text, height=300, key="output_area")

        if output_text:
            st.components.v1.html(
                f"""
                <style>
                    .copy-button {{
                        padding: 12px 24px;  /* ボタンの大きさ調整 */
                        font-size: 16px;  /* フォントサイズ */
                        background-color: #007BFF;
                        color: white;
                        border: none;
                        border-radius: 5px;
                        cursor: pointer;
                        transition: 0.3s;
                    }}
                    .copy-button:hover {{
                        background-color: #0056b3;
                    }}
                    .copy-container {{
                        display: flex;
                        align-items: center;
                        margin-top: 10px;
                    }}
                    .copy-message {{
                        margin-left: 15px;
                        color: green;
                        font-weight: bold;
                        font-size: 16px;
                    }}
                </style>

                <div class="copy-container">
                    <button class="copy-button" onclick="copyToClipboard()">Copy to Clipboard</button>
                    <span id="copyMessage" class="copy-message"></span>
                </div>

                <script>
                function copyToClipboard() {{
                    var text = {repr(output_text)};
                    var tempInput = document.createElement("textarea");
                    tempInput.value = text;
                    document.body.appendChild(tempInput);
                    tempInput.select();
                    document.execCommand("copy");
                    document.body.removeChild(tempInput);

                    var message = document.getElementById("copyMessage");
                    message.innerText = "Copied! ✅";
                    setTimeout(function() {{
                        message.innerText = "";
                    }}, 2000);
                }}
                </script>
                """,
                height=60
            )

if __name__ == "__main__":
    main()
