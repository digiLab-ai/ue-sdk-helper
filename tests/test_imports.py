def test_import_package():
    import uncertainty_engine_helper as ue
    assert hasattr(ue, "__version__")
