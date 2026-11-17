from llm_lab.hello import describe_dependencies, describe_environment


def test_describe_environment_has_python_version():
    info = describe_environment()
    assert "python" in info
    assert info["python"].count(".") >= 1

def test_describe_dependencies():
    result = describe_dependencies()

    assert isinstance(result, str)
    assert result != ""
    assert "." in result

def test_describe_dependencies_when_package_missing():
    result = describe_dependencies("llm-lab-definitely-not-installed-12345")

    assert result is None