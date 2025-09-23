def test_import_package():
    import ue_helper as ue
    assert hasattr(ue, "__version__")
