import json

LANG_DICT = json.load(open("data/raw/marc_lang_codes.json", "r"))
P5_ROOT = (
    "G:/DigiSchol/Digital Research and Curator Team/Projects & Proposals/00_Current Projects"
    "/LibCrowds Convert-a-Card (Adi)/OCR/20230504 TKB Export P5 175 GT pp/1016992/P5_for_Transkribus"
)



# import s3fs

# s3 = s3fs.S3FileSystem(anon=False)
#
#
# @st.cache_data
# def load_s3(s3_path):
#     with s3.open(s3_path, 'rb') as f:
#         df = pickle.load(f)
#         # st.write("Cards data loaded from S3")
#     return df

# cards_df = load_s3('cac-bucket/401_cards.p')