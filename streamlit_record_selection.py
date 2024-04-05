"""
Removed any data processing prior to delivery of cards_df to simplify env for streamlit
Will need to prepare elsewhere then pull in as pickle or csv
"""
import os
import pickle

import pandas as pd
import streamlit as st
from PIL import Image
import s3fs

import cfg
from src.utils import streamlit_utils as st_utils

FANCY_SELECT = False

st.title("Worldcat results for searches for catalogue card title/author")

with open("sidebar_docs.txt", encoding="utf-8") as f:
    sidebar_docs_txt = f.read()
with st.sidebar:
    st.markdown(sidebar_docs_txt)

if os.path.exists("data/processed/401_cards.p"):
    cards_df = pickle.load(open("data/processed/401_cards.p", "rb"))
    st.write("Loaded cards info from local")
else:
    s3 = s3fs.S3FileSystem(anon=False)

    @st.cache_data
    def load_s3(s3_path):
        with s3.open(s3_path, 'rb') as f:
            df = pickle.load(f)
        return df

    cards_df = load_s3('cac-bucket/401_cards.p')
    st.write("Loaded cards info from AWS")

nulls = len(cards_df) - len(cards_df.dropna(subset="worldcat_matches_subtyped"))
cards_to_show = cards_df.dropna(subset="worldcat_matches_subtyped").copy()
cards_to_show.insert(loc=0, column="card_id", value=range(1, len(cards_to_show) + 1))

st.write(f"Showing {len(cards_to_show)} cards with Worldcat results out of of {len(cards_df)} total cards, "
         f"omitting {nulls} without results.")

select_c1, select_c2 = st.columns([0.4, 0.6])
selected_card = select_c1.number_input(
    "Select a card to match",
    min_value=1, max_value=len(cards_to_show),
    help="Type or use +/-"
)

card_table_container = st.empty()

if FANCY_SELECT:
    cards_to_show_selections = st_utils.insert_select_col(cards_to_show)
    subset = ("card_id", "select", "title", "author", "selected_match_ocn", "match_needs_editing", "shelfmark", "lines")
    editable_df, card_selection = st_utils.create_editable_df(card_table_container, cards_to_show_selections, subset)
else:
    subset = ("card_id", "title", "author", "selected_match_ocn", "match_needs_editing", "shelfmark", "lines")
    card_table_container.dataframe(cards_to_show.loc[:, subset], hide_index=True)

# TODO pretty display of column names
card_table_container.dataframe(cards_to_show.loc[:, subset], hide_index=True)
cards_to_show["author"] = cards_df["author"].apply(lambda x: x if x else "")

readable_idx = int(selected_card)
card_idx = cards_to_show.query("card_id == @readable_idx").index.values[0]

MATCH_EXISTS = cards_to_show.loc[card_idx, "selected_match"]
if MATCH_EXISTS:  # U+2800 is a blank character to help centre the green text vertically in the column
    select_c2.write("""\u2800  
                    :green[**This record has already been matched!**]""")

st.write("\n")
st.subheader("Select from Worldcat results")

card_jpg_path = os.path.join("data/raw/chinese/1016992", cards_to_show.loc[card_idx, "xml"][:-5] + ".jpg")

search_ti = cards_to_show.loc[card_idx, 'title'].replace(' ', '+')
search_au = cards_to_show.loc[card_idx, 'author'].replace(' ', '+')
search_term = f"https://www.worldcat.org/search?q=ti%3A{search_ti}+AND+au%3A{search_au}"

ic_left, ic_centred, ic_right = st.columns([0.3, 0.6, 0.1])
ic_centred.image(Image.open(card_jpg_path), use_column_width=True)
label_text = f"""**Right**: Catalogue card  
                 You can check the [Worldcat search]({search_term}) for this card"""
ic_left.write(label_text)

marc_table = st.empty()
check = list(cards_to_show.loc[card_idx, "worldcat_matches_subtyped"])
match_df = pd.DataFrame({"record": list(cards_to_show.loc[card_idx, "worldcat_matches_subtyped"])})
match_df = st_utils.create_filter_columns(match_df, cfg.LANG_DICT, search_au)
all_marc_fields = sorted(list(set(match_df["record"].apply(lambda x: [y.tag for y in x.get_fields()]).sum())))

# Filters form
_header, apply_filters_text = st.columns([0.01, 0.99])
_header.header(body="", anchor="filters")
apply_filters_text.write("Click 'Apply filters' at bottom of box to apply filters")
with st.form("filters"):
    max_to_display_col, removed_records_col = st.columns([0.3, 0.7])

    max_to_display_help = """
    Select the number of records to display in the MARC table above.  
    Setting this value very high can lead to lots of mostly blank rows to scroll through.
    """
    max_to_display = int(
        max_to_display_col.number_input("Max records to display", min_value=1, value=5, help=max_to_display_help))

    # It is possible to remove a previously selected record from the comparison
    records_to_ignore = removed_records_col.multiselect(
        label="Select incorrect records you'd like to remove from the comparison",
        options=match_df.index
    )

    lang_select = st.multiselect(
        "Select Cataloguing Language (040 $b)",
        match_df["language"].unique(),
        format_func=lambda x: f"{x} ({len(match_df.query('language == @x'))} total)",
        default="English"
    )

    st.write("####")
    _, date_slider_col, _ = st.columns([0.05, 0.9, 0.05])
    date_slider = date_slider_col.select_slider(
        label='Select publication year',
        options=match_df.query("publication_date > -9999")["publication_date"].sort_values().dropna().unique().astype(
            int),
        value=(
            match_df.query("publication_date > -9999")["publication_date"].min(), match_df["publication_date"].max()
        ),
        help=("Records with no publication date will remain included in the MARC table. "
              "All records including records with no publication date are included by default when the sliders are in their default end positions. "
              "Publication year defined as a 4-digit number in 260$c")
    )

    st.write("####")
    generic_field_col, generic_field_contains_col, include_recs_without_field_col = st.columns([0.3, 0.475, 0.225])
    search_on_marc_fields = generic_field_col.multiselect(
        "Select MARC field",
        all_marc_fields,
        help="[LoC MARC fields](https://www.loc.gov/marc/bibliographic/)"
    )
    search_terms = generic_field_contains_col.text_input(
        "MARC field contains",
        help=("For multiple fields separate terms by a semi-colon. "
              "e.g. if specifying fields 010, 300 then search term might be '2001627090; 140 pages'."
              "Searching on a field with repeat fields searches all the repeat fields"
              )
    )
    search_terms = search_terms.split(";")
    if search_terms == [""]: search_terms = []
    include_recs_without_field = include_recs_without_field_col.checkbox("Allow records without specified MARC fields")

    if len(search_on_marc_fields) != len(search_terms):
        st.markdown(
            (f":red[**Searching on {len(search_on_marc_fields)} MARC fields, "
             f"but {len(search_terms.split(';'))} search terms specified. "
             f"Please change number of searched on MARC fields or number of ';' seperated search terms**]")
        )

    # filter option columns defined below to display in the filters users can choose from
    filter_options = ["num_subject_access", "num_rda", "num_linked", "has_phys_desc", "good_encoding_level",
                      "record_length"]

    sort_options_col, highlight_col, apply_col = st.columns([0.65, 0.2, 0.15])
    sort_options = sort_options_col.multiselect(
        label=(
            "Select how to sort matching records."
        ),
        options=filter_options,
        format_func=st_utils.pretty_filter_option,
        help=("The default is the order in which results are returned from Worldcat."
              "If more than one option is selected results will be sorted sequentially in the order options have been selected.")
    )

    highlight_button = highlight_col.checkbox("Highlight common fields", value=True,
                                              help="Highlight field values that are common between two or more records.")

    apply_filters = apply_col.form_submit_button(
        label="Apply filters"
    )

filter_query = "language in @lang_select & ((@date_slider[0] <= publication_date and publication_date <= @date_slider[1]) or publication_date == -9999)"
filtered_df = match_df.query(filter_query).copy()
sorted_filtered_df = filtered_df.sort_values(by=sort_options, ascending=False)

formatted_records, fmt_new_idx = [], []
for i in range(len(sorted_filtered_df)):
    res = sorted_filtered_df.iloc[i, 0].get_fields()
    ldr = sorted_filtered_df.iloc[i, 0].leader
    col = pd.DataFrame(
        index=pd.Index(["LDR"] + [x.tag for x in res], name="Field"),
        data=[ldr] + [x.__str__()[6:] for x in res],
        columns=[sorted_filtered_df.iloc[i].name]
    )
    formatted_records.append(st_utils.gen_unique_idx(col))
    fmt_new_idx.append(st_utils.gen_sf_rpt_unique_idx(col))

marc_table_all_recs_df = pd.concat(formatted_records, axis=1).sort_index(key=st_utils.sort_fields_idx)
new_marc_table = pd.concat(fmt_new_idx, axis=1).sort_index()
st_utils.simplify_6xx(new_marc_table)

marc_table_filtered_recs = st_utils.filter_on_generic_fields(marc_table_all_recs_df, search_on_marc_fields,
                                                             search_terms, include_recs_without_field)
match_ids = marc_table_filtered_recs.columns.tolist()

record = "records"
if len(records_to_ignore) == 1: record = "record"

filtered_records_text = f"""
[Max records to display](#filters) set to {max_to_display}. Displaying {max_to_display} of {len(match_ids)} filtered records.\n
{len(match_df)} total records.  
{len(match_df) - len(match_ids)} removed by filters.  
{len(records_to_ignore)} incorrect {record} removed by user.
"""
ic_left.write(filtered_records_text)

records_to_display = [x for x in match_ids if x not in records_to_ignore]
marc_table_df = marc_table_all_recs_df.loc[:, records_to_display[:max_to_display]].dropna(how="all")
marc_table.dataframe(st_utils.style_marc_df(marc_table_df, highlight_button, MATCH_EXISTS))

with st.form("record_selection"):
    closest_result_col, editing_required_col, save_col = st.columns(3)
    selected_match = closest_result_col.radio(
        label="Which is the closest Worldcat result?",
        options=(records_to_display[:max_to_display] + ["No correct results"])
    )
    needs_editing = editing_required_col.radio(
        label="Does this record need manual editing or is it ready to ingest?",
        options=[True, False],
        format_func=lambda x: {True: "Manual editing", False: "Ready to ingest"}[x]
    )
    save_res = save_col.form_submit_button(
        label="Save selection"
    )
    clear_res = save_col.form_submit_button(
        label="Clear selection"
    )

    if save_res:
        if selected_match == "None of the results are correct":
            cards_df.loc[card_idx, ["selected_match", "selected_match_ocn"]] = "No matches"
        else:
            cards_df.loc[card_idx, "selected_match"] = selected_match
            cards_df.loc[card_idx, "selected_match_ocn"] = \
                cards_df.loc[card_idx, "worldcat_matches_subtyped"][selected_match].get_fields("001")[0].data
            cards_df.loc[card_idx, "match_needs_editing"] = needs_editing

        # with s3.open('cac-bucket/cards_df.p', 'wb') as f:
        #     pickle.dump(cards_df, f)
        pickle.dump(cards_df, open("notebooks/401_cards.p", "wb"))
        st.cache_data.clear()  # Needed if pulling from S3
        st_utils.update_card_table(cards_df, subset, card_table_container, FANCY_SELECT)
        st_utils.update_marc_table(marc_table, marc_table_df, highlight_button, MATCH_EXISTS)
        st.markdown("### Selection saved!")

    if clear_res:
        cards_df.loc[card_idx, ["selected_match", "selected_match_ocn", "match_needs_editing"]] = None
        # with s3.open('cac-bucket/cards_df.p', 'wb') as f:
        #     pickle.dump(cards_df, f)
        pickle.dump(cards_df, open("notebooks/401_cards.p", "wb"))
        st.cache_data.clear()  # Needed if pulling from S3
        st_utils.update_card_table(cards_df, subset, card_table_container, FANCY_SELECT)
        st.markdown("### Selection cleared!")
