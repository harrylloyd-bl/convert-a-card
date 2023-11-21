"""
Removed any data processing prior to delivery of cards_df to simplify env for streamlit
Will need to prepare elsewhere then pull in as pickle or csv
"""
import re
import os
import pickle
import json
from random import random

from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
import s3fs

# s3 = s3fs.S3FileSystem(anon=False)
#
#
# @st.cache_data
# def load_s3(s3_path):
#     with s3.open(s3_path, 'rb') as f:
#         df = pickle.load(f)
#         # st.write("Cards data loaded from S3")
#     return df

FANCY_SELECT = False

st.title("Worldcat results for searches for catalogue card title/author")

if 'stale' not in st.session_state:
    st.session_state['stale'] = False
elif st.session_state['stale']:
    st.session_state['stale'] = False

# cards_df = load_s3('cac-bucket/401_cards.p')
cards_df = pickle.load(open("notebooks/401_cards.p", "rb"))
cards_df = cards_df.iloc[:175].copy()  # just while we can't access the network drives
nulls = len(cards_df) - len(cards_df.dropna(subset="worldcat_matches_subtyped"))
cards_to_show = cards_df.dropna(subset="worldcat_matches_subtyped").copy()
cards_to_show.insert(loc=0, column="card_id", value=range(1, len(cards_to_show) + 1))

if FANCY_SELECT:
    cards_to_show_selections = cards_to_show.copy()
    cards_to_show_selections.insert(loc=1, column="select", value=False)
    if st.session_state.get("selected_card"):
        st.write(f"{st.session_state.get('selected_card')}, {type(st.session_state.get('selected_card'))}")
        cards_to_show_selections.iloc[st.session_state.get("selected_card"), 1] = True
    subset = ("card_id", "select", "title", "author", "selected_match_ocn", "match_needs_editing", "shelfmark", "lines")
else:
    subset = ("card_id", "title", "author", "selected_match_ocn", "match_needs_editing", "shelfmark", "lines")

st.write(f"Showing {len(cards_to_show)} cards with Worldcat results out of of {len(cards_df)} total cards, "
         f"omitting {nulls} without results.")


select_c1, select_c2 = st.columns([0.4, 0.6])
selected_card = select_c1.number_input(
    "Select a card to match",
    min_value=1, max_value=len(cards_to_show),
    help="Type or use +/-"
)


def update_card_table(card_table_container):
    cards_to_show = cards_df.dropna(subset="worldcat_matches_subtyped")
    cards_to_show.insert(loc=0, column="card_id", value=range(1, len(cards_to_show) + 1))
    if not FANCY_SELECT:
        card_table_container.dataframe(cards_to_show.loc[:, subset], hide_index=True)

    else:
        st.session_state.pop("card_table")
        cards_to_show.insert(loc=0, column="card_id", value=range(1, len(cards_to_show) + 1))
        cards_to_show_selections = cards_to_show.copy()
        cards_to_show_selections.insert(loc=1, column="select", value=False)
        if st.session_state["selected_card"]:
            cards_to_show_selections.iloc[st.session_state["selected_card"], 1] = True
        st.write("card_table_1")
        card_table_container.data_editor(
            cards_to_show_selections.loc[:, subset],
            hide_index=True,
            column_config={"select": st.column_config.CheckboxColumn(required=True)},
            disabled=cards_to_show.columns,
            on_change=table_needs_refresh("card_table_1"),
            key="card_table_1"
        )


def table_needs_refresh(key):
    # key is card_table, the key of the df this is the on_change fn for
    st.write(f"key: {key}")
    if "selected_card" not in st.session_state:
        st.session_state["selected_card"] = None
    if key not in st.session_state:
        st.write("card_table not yet in session state")
        return None

    # clear out any edited_rows that are False/empty so they don't affect len("edited_rows") calculations
    st.write("total inc false: " + f"{len(st.session_state[key]['edited_rows'])}")
    iter_copy = st.session_state[key]["edited_rows"].copy()
    for k, v in iter_copy.items():
        if not v["select"]:
            st.session_state[key]["edited_rows"].pop(k)
    st.write("total true: " + f"{len(st.session_state[key]['edited_rows'])}")

    st.write("refreshing table")
    st.session_state[key]

    if len(st.session_state[key]["edited_rows"]) == 0:
        st.write("no edited rows")

    if len(st.session_state[key]["edited_rows"]) == 1:
        st.session_state["selected_card"] = [int(x) for x in st.session_state[key]["edited_rows"].keys()][0]
        st.session_state["selected_card"]
        st.write("one edited rows")

    if len(st.session_state[key]["edited_rows"]) == 2:
        all_edited = [int(x) for x in st.session_state[key]["edited_rows"].keys()]
        old_card = all_edited.pop(st.session_state["selected_card"])
        new_card = all_edited[0]
        st.write(new_card)
        st.session_state["selected_card"] = new_card
        st.session_state[key]["edited_rows"] = {f"{new_card}": {"select": True}}
        st.write("updated state:")
        st.session_state[key]
        # update_card_table(card_table_container)
        st.write("two edited rows")

    if len(st.session_state[key]["edited_rows"]) > 2:  # too many cards clicked at once
        all_edited = [int(x) for x in st.session_state[key]["edited_rows"].keys()]
        old_card = all_edited.pop(st.session_state["selected_card"])
        new_card = all_edited[0]  # pick the lowest idx of the ones the user has clicked, arbitrary choice
        st.session_state["selected_card"] = new_card
        st.session_state[key]["edited_rows"] = {f"{new_card}": {"select": True}}
        st.write("more than 2 edited rows")

    if st.session_state["selected_card"] == "hello":
        if len(card_selection) == 0:
            st.write("len = 0")
            st.session_state["selected_card"] = None
            st.write(st.session_state["selected_card"])
            if st.session_state["stale"]:
                st.write("stale")
                # update_card_table(card_table_container)

        elif len(card_selection) == 1:
            st.write("len = 1")
            st.write(card_selection)
            st.session_state["selected_card"] = card_selection[-1]
            if st.session_state["stale"]:
                st.write("stale")
                # update_card_table(card_table_container)

        elif len(card_selection) == 2:
            st.write("len = 2")
            old_card = st.session_state["selected_card"]
            st.write(f"old_card {old_card}")
            st.write(card_selection)
            selection = card_selection
            selection.remove(old_card)
            st.session_state["selected_card"] = selection[0]
            # TODO fix references to editable_df - what should these be references to? function arg?
            editable_df.loc[old_card, "select"] = False
            st.write(editable_df[editable_df["select"]].drop("select", axis=1))
            if st.session_state["stale"]:
                st.write("stale")
                # update_card_table(card_table_container)
        #
        # elif len(card_selection) > 2:  # someone's clicked too many cards, take the highest val one they've clicked
        #     st.write("len > 2")
        #     old_card = st.session_state["selected_card"]
        #     all_selected = card_selection
        #     all_selected.remove(old_card)
        #     st.session_state["selected_card"] = all_selected[-1]
        #     editable_df.loc[all_selected[:-1] + [old_card], "select"] = False
        #     st.write(editable_df[editable_df["select"]].drop("select", axis=1))
        #     if st.session_state["stale"]:
        #         update_card_table()

card_table_container = st.empty()

if FANCY_SELECT:
    # https://docs.streamlit.io/knowledge-base/using-streamlit/how-to-get-row-selections
    editable_df = card_table_container.data_editor(
        cards_to_show_selections.loc[:, subset],
        hide_index=True,
        column_config={"select": st.column_config.CheckboxColumn(required=True)},
        disabled=cards_to_show.columns,
        on_change=table_needs_refresh("card_table"),
        key="card_table"
    )
    st.write("after table declaration")
    card_selection = editable_df[editable_df["select"]].drop("select", axis=1).index.to_list()

    if 'selected_card' not in st.session_state:
        st.session_state['selected_card'] = None

else:
    card_table_container.dataframe(cards_to_show.loc[:, subset], hide_index=True)

# TODO pretty display of column names
# TODO enforce that when the df is saved it's saved with all 0s in the select_card column
card_table_container.dataframe(cards_to_show.loc[:, subset], hide_index=True)
cards_to_show["author"] = cards_df["author"].apply(lambda x: x if x else "")

readable_idx = int(selected_card)
card_idx = cards_to_show.query("card_id == @readable_idx").index.values[0]

existing_selected_match = cards_to_show.loc[card_idx, "selected_match"]
if existing_selected_match:  # U+2800 is a blank character to help centre the green text vertically in the column
    select_c2.write("""\u2800  
                    :green[**This record has already been matched!**]""")

st.write("\n")
st.subheader("Select from Worldcat results")

# p5_root = (
#     "G:/DigiSchol/Digital Research and Curator Team/Projects & Proposals/00_Current Projects"
#     "/LibCrowds Convert-a-Card (Adi)/OCR/20230504 TKB Export P5 175 GT pp/1016992/P5_for_Transkribus"
# )

card_jpg_path = os.path.join("data/images", cards_to_show.loc[card_idx, "xml"][:-5] + ".jpg")

search_ti = cards_to_show.loc[card_idx, 'title'].replace(' ', '+')
search_au = cards_to_show.loc[card_idx, 'author'].replace(' ', '+')
search_term = f"https://www.worldcat.org/search?q=ti%3A{search_ti}+AND+au%3A{search_au}"

ic_left, ic_centred, ic_right = st.columns([0.3,0.6,0.1])
ic_centred.image(Image.open(card_jpg_path), use_column_width=True)
label_text = f"""
**Right**: Catalogue card  
You can check the [Worldcat search]({search_term}) for this card
"""
ic_left.write(label_text)

marc_table = st.empty()
match_df = pd.DataFrame({"record": list(cards_to_show.loc[card_idx, "worldcat_matches_subtyped"])})

# filter options
match_df["has_title"] = match_df["record"].apply(lambda x: bool(x.get_fields("245")))
match_df["has_author"] = match_df["record"].apply(lambda x: bool(x.get_fields("100", "110", "111", "130")))
au_exists = bool(search_au)
match_df = match_df.query("has_title == True and (has_author == True or not @au_exists)")

lang_dict = json.load(open("data/raw/marc_lang_codes.json", "r"))

re_040b = re.compile(r"\$b[a-z]+\$")
match_df["language_040$b"] = match_df["record"].apply(lambda x: re_040b.search(x.get_fields("040")[0].__str__()).group())
match_df["language"] = match_df["language_040$b"].str[2:-1].map(lang_dict["codes"])

# filter option columns defined below to display in the filters users can choose from
filter_options = ["num_subject_access", "num_rda", "num_linked", "has_phys_desc", "good_encoding_level", "record_length"]

subject_access_fields = ["600", "610", "611", "630", "647", "648", "650", "651", "653", "654", "655", "656", "657", "658", "662", "688"]
rda_fields = ["264", "336", "337", "338", "344", "345", "346", "347"]
match_df["num_subject_access"] = match_df["record"].apply(lambda x: len(x.get_fields(*subject_access_fields)))
match_df["num_rda"] = match_df["record"].apply(lambda x: len(x.get_fields(*rda_fields)))

match_df["num_linked"] = match_df["record"].apply(lambda x: len(x.get_fields("880")))
match_df["has_phys_desc"] = match_df["record"].apply(lambda x: bool(x.get_fields("300")))
match_df["good_encoding_level"] = match_df["record"].apply(lambda x: x.leader[17] not in [3, 5, 7])
match_df["record_length"] = match_df["record"].apply(lambda x: len(x.get_fields()))

def get_pub_date(record):
    # Look for a date in first 260$c, if absent include in search anyway
    f260 = record.get_fields("260")
    c_subfields = []
    for x in f260:
        c_subfields.extend(x.get_subfields("c"))
    re_260c = re.compile(r"[0-9]{4,4}")
    if c_subfields:
        match = re_260c.search(c_subfields[0])
        if match:
            return int(match.group())
        else:
            return -9999
    else:
        return -9999


match_df["publication_date"] = match_df["record"].apply(lambda x: get_pub_date(x))

all_marc_fields = sorted(list(set(match_df["record"].apply(lambda x: [y.tag for y in x.get_fields()]).sum())))

def pretty_filter_option(option):
    display_dict = {
        "num_subject_access": "Number of subject access fields",
        "num_rda": "Number of RDA fields",
        "num_linked": "Number of linked fields",
        "has_phys_desc": "Has a physical description",
        "good_encoding_level": "Encoding level not 3/5/7",
        "record_length": "Number of fields in record"
    }
    return display_dict[option]

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
        options=match_df.query("publication_date > -9999")["publication_date"].sort_values().dropna().unique().astype(int),
        value=(match_df.query("publication_date > -9999")["publication_date"].min(), match_df["publication_date"].max()),
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

    sort_options_col, _, apply_col = st.columns([0.7, 0.05, 0.25])
    sort_options = sort_options_col.multiselect(
        label=(
            "Select how to sort matching records. The default is the order the results are returned from Worldcat."
            " Results will be sorted in the order options are selected"
        ),
        options=filter_options,
        format_func=pretty_filter_option
    )

    apply_col.write("####")
    apply_filters = apply_col.form_submit_button(
        label="Apply filters"
    )

filtered_df = match_df.query(
    ("language in @lang_select & ((@date_slider[0] <= publication_date and publication_date <= @date_slider[1]) or publication_date == -9999)")
).copy()

sorted_filtered_df = filtered_df.sort_values(
    by=sort_options,
    ascending=False
)

def gen_unique_idx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a unique index from one that contains repeated fields
    @param df: pd.DataFrame
    @return: pd.DataFrame
    """
    df["Repeat Field ID"] = ""
    dup_idx = df.index[df.index.duplicated()].unique()
    unhandled_fields = [x for x in dup_idx if x not in ["650", "880"]]
    if "650" in dup_idx:
        str_add = df.loc["650", df.columns[0]].copy()
        str_add = [" " + str(x) for x in range(len(str_add))]
        df.loc["650", "Repeat Field ID"] = df.loc["650", df.columns[0]].str.split(" ").transform(lambda x: x[0]) + str_add
    if "880" in dup_idx:
        str_add = df.loc["880", df.columns[0]].copy()
        str_add = [" " + str(x) for x in range(len(str_add))]
        df.loc["880", "Repeat Field ID"] = df.loc["880", df.columns[0]].str.split("/").transform(lambda x: x[0]) + str_add
    for dup in unhandled_fields:
        df.loc[dup, "Repeat Field ID"] = [str(x) for x in range(len(df.loc[dup]))]

    return df.set_index("Repeat Field ID", append=True)


def sort_fields_idx(index: pd.Index) -> pd.Index:
    """
    Specific keys to sort indices containing MARC fields
    @param index: pd.Index
    @return: pd.Index
    """
    if index.name == "MARC Field":
        key = [0 if x == "LDR" else int(x) for x in index]
        return pd.Index(key)
    elif index.name == "Repeat Field ID":
        key = [x.split("$")[1] if "$" in x else x for x in index]
        return pd.Index(key)


displayed_matches = []
for i in range(len(sorted_filtered_df)):
    res = sorted_filtered_df.iloc[i, 0].get_fields()
    ldr = sorted_filtered_df.iloc[i, 0].leader
    col = pd.DataFrame(
        index=pd.Index(["LDR"] + [x.tag for x in res], name="MARC Field"),
        data=[ldr] + [x.__str__()[6:] for x in res],
        columns=[sorted_filtered_df.iloc[i].name]
    )
    displayed_matches.append(gen_unique_idx(col))

marc_table_all_recs_df = pd.concat(displayed_matches, axis=1).sort_index(key=sort_fields_idx)

def filter_on_generic_fields(df, fields, terms, include_recs_without_field):
    """
    Input the marc_table_df with all records
    sum all repeat columns and transpose to make searching easier
    Search each column in fields for the corresponding term in terms
    Return a df with only records that match the search terms
    @param df:
    @param fields:
    @param terms:
    @return:
    """
    if not fields or not terms:
        return df

    t_df = df.groupby(level=0).sum().T
    terms = [x.strip() for x in terms]
    filter_df = pd.concat([t_df[field].str.contains(term) for field, term in zip(fields, terms)], axis=1)
    if include_recs_without_field:
        filter = filter_df.all(axis=1)
    else:
        filter = filter_df.where(lambda x: ~x.isna(), False).all(axis=1)
    return df.T[filter].T

marc_table_filtered_recs = filter_on_generic_fields(marc_table_all_recs_df, search_on_marc_fields, search_terms, include_recs_without_field)
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


def gen_gmap(col):
    counts = col.value_counts()
    to_highlight = counts[counts > 1]
    no_highlight = counts[counts == 1]
    colour_vals = np.linspace(0, 1, len(to_highlight) + 2)[1:-1]
    mapping = {k:v for k,v in zip(to_highlight.index, colour_vals)}
    for val in no_highlight.index:
        mapping[val] = -10
    return col.map(mapping, na_action='ignore')


def style_marc_df(df):
    gmap = df.apply(gen_gmap, axis=1)
    gmap[gmap.isna()] = -10
    gmap[1::3] += 0.05
    gmap[2::3] += 0.1
    styled_df = df.style.background_gradient(gmap=gmap, vmin=0, vmax=1, axis=None)
    if existing_selected_match and existing_selected_match in df.columns:
        styled_df = styled_df.highlight_between(subset=[existing_selected_match], color="#2FD033A0")
    return styled_df


records_to_display = [x for x in match_ids if x not in records_to_ignore]
marc_table_df = marc_table_all_recs_df.loc[:, records_to_display[:max_to_display]].dropna(how="all")
marc_table.dataframe(style_marc_df(marc_table_df))

def update_marc_table(table, df):
    return table.dataframe(style_marc_df(df))

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
            cards_df.loc[card_idx, "selected_match_ocn"] = cards_df.loc[card_idx, "worldcat_matches_subtyped"][selected_match].get_fields("001")[0].data
            cards_df.loc[card_idx, "match_needs_editing"] = needs_editing

        # with s3.open('cac-bucket/cards_df.p', 'wb') as f:
        #     pickle.dump(cards_df, f)
        pickle.dump(cards_df, open("notebooks/401_cards.p", "wb"))
        st.cache_data.clear()  # Needed if pulling from S3
        update_card_table(card_table_container)
        update_marc_table(marc_table, marc_table_df)
        st.markdown("### Selection saved!")

    if clear_res:
        cards_df.loc[card_idx, ["selected_match", "selected_match_ocn", "match_needs_editing"]] = None
        # with s3.open('cac-bucket/cards_df.p', 'wb') as f:
        #     pickle.dump(cards_df, f)
        pickle.dump(cards_df, open("notebooks/401_cards.p", "wb"))
        st.cache_data.clear()  # Needed if pulling from S3
        update_card_table(card_table_container)
        st.markdown("### Selection cleared!")
