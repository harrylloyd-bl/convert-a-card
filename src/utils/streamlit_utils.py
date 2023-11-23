import re

import numpy as np
import pandas as pd
import streamlit as st


def get_pub_date(record):
    # Look for a date in first 260$c, if absent include in search anyway
    f260 = record.get_fields("260")
    c_subfields = []
    for x in f260:
        c_subfields.extend(x.get_subfields("c"))
    re_260c = re.compile(r"[0-9]{4}")
    if c_subfields:
        match = re_260c.search(c_subfields[0])
        if match:
            return int(match.group())
        else:
            return -9999
    else:
        return -9999


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


def gen_unique_idx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a unique index from one that contains repeated fields
    @param df: pd.DataFrame
    @return: pd.DataFrame
    """
    out_df = df.copy()
    out_df["Repeat Field ID"] = ""
    dup_idx = out_df.index[out_df.index.duplicated()].unique()
    unhandled_fields = [x for x in dup_idx if x not in ["650", "880"]]
    if "650" in dup_idx:
        str_add = out_df.loc["650", out_df.columns[0]].copy()
        str_add = [" " + str(x) for x in range(len(str_add))]
        out_df.loc["650", "Repeat Field ID"] = out_df.loc["650", out_df.columns[0]].str.split(" ").transform(
            lambda x: x[0]) + str_add
    if "880" in dup_idx:
        str_add = out_df.loc["880", out_df.columns[0]].copy()
        str_add = [" " + str(x) for x in range(len(str_add))]
        out_df.loc["880", "Repeat Field ID"] = out_df.loc["880", out_df.columns[0]].str.split("/").transform(
            lambda x: x[0]) + str_add
    for dup in unhandled_fields:
        out_df.loc[dup, "Repeat Field ID"] = [str(x) for x in range(len(out_df.loc[dup]))]

    return out_df.set_index("Repeat Field ID", append=True)


def simplify_6xx(df):
    """
    Pandas magic
    Arbitrarily assinging repeat_id vals to repeat fields in records means common field values are not matched to each other
    Get round this by reindexing on an index of all unique values for a subfield
    Set the values for the reindexed subfield to the newly reindexed one.
    If there has been reordering this leaves rows of NA at the end of the subfield that can be dropped
    """
    df = df.copy()
    subfields = df.loc["650"].index.get_level_values(0)
    for sf in subfields:
        sf_orig = df.loc[("650", sf), :]
        sf_unique_vals = pd.Series(sf_orig.values.flatten()).dropna().unique()
        sf_unique_df = pd.DataFrame(data=sf_unique_vals, columns=pd.Index(["unique_vals"]))
        for x in df.columns:
            sf_unique_df = sf_unique_df.merge(sf_orig[x], how="left", left_on="unique_vals", right_on=x)
        replacement_df = sf_unique_df.set_index(sf_orig.index[:len(sf_unique_df)]).reindex(sf_orig.index).drop(
            columns="unique_vals")
        replacement_df["Field"] = "650"
        replacement_df["Subfield"] = "\\7"
        replacement_df = replacement_df.reindex(["Field", "Subfield"], append=True).reorder_levels([1, 2, 0])
        df.loc[("650", sf), :] = replacement_df
    return df


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
        df_filter = filter_df.all(axis=1)
    else:
        df_filter = filter_df.where(lambda x: ~x.isna(), False).all(axis=1)
    return df.T[df_filter].T


def gen_gmap(col):
    counts = col.value_counts()
    to_highlight = counts[counts > 1]
    no_highlight = counts[counts == 1]
    colour_vals = np.linspace(0, 1, len(to_highlight) + 2)[1:-1]
    mapping = {k: v for k, v in zip(to_highlight.index, colour_vals)}
    for val in no_highlight.index:
        mapping[val] = -10
    return col.map(mapping, na_action='ignore')


def style_marc_df(df, highlight_common_vals, existing_selected_match):
    styled_df = df.style
    if highlight_common_vals:
        gmap = df.apply(gen_gmap, axis=1)
        gmap[gmap.isna()] = -10
        gmap[1::3] += 0.05
        gmap[2::3] += 0.1
        styled_df = styled_df.background_gradient(gmap=gmap, vmin=0, vmax=1, axis=None)
    if existing_selected_match and existing_selected_match in df.columns:
        styled_df = styled_df.highlight_between(subset=[existing_selected_match], color="#2FD033A0")
    return styled_df


def create_filter_columns(df, lang_dict, search_au):
    df["has_title"] = df["record"].apply(lambda x: bool(x.get_fields("245")))
    df["has_author"] = df["record"].apply(lambda x: bool(x.get_fields("100", "110", "111", "130")))
    au_exists = bool(search_au)
    df = df.query("has_title == True and (has_author == True or not @au_exists)")
    re_040b = re.compile(r"\$b[a-z]+\$")
    df["language_040$b"] = df["record"].apply(
        lambda x: re_040b.search(x.get_fields("040")[0].__str__()).group())
    df["language"] = df["language_040$b"].str[2:-1].map(lang_dict["codes"])

    subject_access_fields = ["600", "610", "611", "630", "647", "648", "650", "651", "653", "654", "655", "656", "657",
                             "658", "662", "688"]
    rda_fields = ["264", "336", "337", "338", "344", "345", "346", "347"]
    df["num_subject_access"] = df["record"].apply(lambda x: len(x.get_fields(*subject_access_fields)))
    df["num_rda"] = df["record"].apply(lambda x: len(x.get_fields(*rda_fields)))

    df["num_linked"] = df["record"].apply(lambda x: len(x.get_fields("880")))
    df["has_phys_desc"] = df["record"].apply(lambda x: bool(x.get_fields("300")))
    df["good_encoding_level"] = df["record"].apply(lambda x: x.leader[17] not in [3, 5, 7])
    df["record_length"] = df["record"].apply(lambda x: len(x.get_fields()))
    df["publication_date"] = df["record"].apply(lambda x: get_pub_date(x))

    return df


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


def update_marc_table(table, df, highlight_button, match_exists):
    return table.dataframe(style_marc_df(df, highlight_button, match_exists))


# functions for an interactive selection column
def insert_select_col(df):
    cards_to_show_selections = df.copy()
    cards_to_show_selections.insert(loc=1, column="select", value=False)
    if st.session_state.get("selected_card"):
        st.write(f"{st.session_state.get('selected_card')}, {type(st.session_state.get('selected_card'))}")
        cards_to_show_selections.iloc[st.session_state.get("selected_card"), 1] = True
    return cards_to_show_selections


def create_editable_df(card_table_container, df, subset):
    # https://docs.streamlit.io/knowledge-base/using-streamlit/how-to-get-row-selections
    editable_df = card_table_container.data_editor(
        df.loc[:, subset],
        hide_index=True,
        column_config={"select": st.column_config.CheckboxColumn(required=True)},
        disabled=df.drop(columns="select").columns,
        on_change=table_needs_refresh("card_table"),
        # now that these are functions need to work out how to pass edited_df back as an arg - have to use st.session_state
        key="card_table"
    )
    st.write("after table declaration")
    card_selection = editable_df[editable_df["select"]].drop("select", axis=1).index.to_list()

    if 'selected_card' not in st.session_state:
        st.session_state['selected_card'] = None

    return editable_df, card_selection


def update_card_table(cards_df, subset, card_table_container, fancy_select=False):
    cards_to_show = cards_df.dropna(subset="worldcat_matches_subtyped")
    cards_to_show.insert(loc=0, column="card_id", value=range(1, len(cards_to_show) + 1))
    if not fancy_select:
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


def table_needs_refresh(key, card_selection, editable_df):
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
