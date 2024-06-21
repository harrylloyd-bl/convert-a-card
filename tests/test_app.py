import sys
from streamlit.testing.v1 import AppTest


def test_no_uncaught_exception():
    try:
        AppTest.from_file("streamlit_record_selection.py", default_timeout=30).run()
    except ValueError as e:
        print(f"Caught {type(e)}, check values passed to widgets")
        print(sys.exception())
        assert False
    except KeyError as e:
        print(f"Caught {type(e)}, check dataframes")
        print(sys.exception())
        assert False