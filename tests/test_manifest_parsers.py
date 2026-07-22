import pytest

from agent.workspace.manifest_parsers import (
    _parse_build_gradle,
    _parse_cargo_toml,
    _parse_csproj,
    _parse_go_mod,
    _parse_package_json,
    _parse_pom_xml,
    _parse_pyproject_toml,
    _parse_requirements_txt,
    _parse_setup_py,
)


class TestParseRequirementsTxt:
    def test_missing_file(self, tmp_path):
        assert _parse_requirements_txt(tmp_path / "requirements.txt") == []

    @pytest.mark.parametrize(
        "content, expected",
        [
            ("", []),
            ("requests>=2.0\n", ["requests"]),
            ("requests[security]>=2.0\n", ["requests"]),
            ("-r other.txt\n-c constraints.txt\n-e git+https://example.com\n", []),
            ("# comment\n\nrequests\n", ["requests"]),
            ("flask\ndjango\nfastapi\n", ["flask", "django", "fastapi"]),
        ],
    )
    def test_parses(self, tmp_path, content, expected):
        f = tmp_path / "requirements.txt"
        f.write_text(content)
        assert _parse_requirements_txt(f) == expected


class TestParsePackageJson:
    def test_missing_file(self, tmp_path):
        assert _parse_package_json(tmp_path / "package.json") == []

    def test_dependencies_and_dev_dependencies(self, tmp_path):
        f = tmp_path / "package.json"
        f.write_text('{"dependencies": {"react": "^18.0"}, "devDependencies": {"typescript": "^5.0"}}')
        result = _parse_package_json(f)
        assert "react" in result
        assert "typescript" in result

    @pytest.mark.parametrize(
        "content, expected",
        [
            ("", []),
            ('{"name": "my-app", "version": "1.0.0"}', []),
            ("{not valid json", []),
        ],
    )
    def test_parses(self, tmp_path, content, expected):
        f = tmp_path / "package.json"
        f.write_text(content)
        assert _parse_package_json(f) == expected


class TestParsePyprojectToml:
    def test_missing_file(self, tmp_path):
        assert _parse_pyproject_toml(tmp_path / "pyproject.toml") == []

    def test_empty_file(self, tmp_path):
        f = tmp_path / "pyproject.toml"
        f.write_text("")
        assert _parse_pyproject_toml(f) == []

    @pytest.mark.parametrize(
        "content, expected1, expected2",
        [
            (
                "[project.dependencies]\nrequests = \">=2.0\"\nflask = \">=2.0\"\n",
                "requests",
                "flask",
            ),
            (
                "[tool.poetry.dependencies]\npython = \"^3.11\"\nhttpx = \"^0.24\"\n",
                "python",
                "httpx",
            ),
        ],
    )
    def test_parses(self, tmp_path, content, expected1, expected2):
        f = tmp_path / "pyproject.toml"
        f.write_text(content)
        result = _parse_pyproject_toml(f)
        assert expected1 in result
        assert expected2 in result

    def test_other_sections_ignored(self, tmp_path):
        f = tmp_path / "pyproject.toml"
        f.write_text("[tool.ruff]\nline-length = \"88\"\n\n[project.dependencies]\nrequests = \">=2.0\"\n")
        result = _parse_pyproject_toml(f)
        assert "line-length" not in result
        assert "requests" in result


class TestParseCargoToml:
    def test_missing_file(self, tmp_path):
        assert _parse_cargo_toml(tmp_path / "Cargo.toml") == []

    def test_empty_file(self, tmp_path):
        f = tmp_path / "Cargo.toml"
        f.write_text("")
        assert _parse_cargo_toml(f) == []

    def test_dependencies_section(self, tmp_path):
        f = tmp_path / "Cargo.toml"
        f.write_text("[dependencies]\nserde = \"1.0\"\ntokio = \"1.0\"\n")
        result = _parse_cargo_toml(f)
        assert "serde" in result
        assert "tokio" in result

    def test_other_sections_ignored(self, tmp_path):
        f = tmp_path / "Cargo.toml"
        f.write_text("[dev-dependencies]\ncriterion = \"0.4\"\n\n[dependencies]\nserde = \"1.0\"\n")
        result = _parse_cargo_toml(f)
        assert "criterion" not in result
        assert "serde" in result

    def test_inline_table_dep(self, tmp_path):
        f = tmp_path / "Cargo.toml"
        f.write_text('[dependencies]\nserde = { version = "1", features = ["derive"] }\n')
        result = _parse_cargo_toml(f)
        assert "serde" in result


class TestParseCsproj:
    def test_missing_file(self, tmp_path):
        assert _parse_csproj(tmp_path / "project.csproj") == []

    @pytest.mark.parametrize(
        "content, expected",
        [
            ("", []),
            (
                '<Project><ItemGroup>'
                '<PackageReference Include="Newtonsoft.Json" Version="13.0" />'
                '</ItemGroup></Project>',
                ["Newtonsoft.Json"],
            ),
            (
                '<Project xmlns="http://schemas.microsoft.com/developer/msbuild/2003">'
                '<ItemGroup>'
                '<PackageReference Include="Serilog" Version="3.0" />'
                '</ItemGroup></Project>',
                ["Serilog"],
            ),
            ('<Project><ItemGroup></ItemGroup></Project>', []),
            ("<Project><ItemGroup>", []),
        ],
    )
    def test_parses(self, tmp_path, content, expected):
        f = tmp_path / "project.csproj"
        f.write_text(content)
        assert _parse_csproj(f) == expected


class TestParseSetupPy:
    def test_missing_file(self, tmp_path):
        assert _parse_setup_py(tmp_path / "setup.py") == []

    def test_install_requires(self, tmp_path):
        f = tmp_path / "setup.py"
        f.write_text(
            'setup(\n'
            '    install_requires=[\n'
            '        "requests",\n'
            '        "flask",\n'
            '    ]\n'
            ')\n'
        )
        result = _parse_setup_py(f)
        assert "requests" in result
        assert "flask" in result

    def test_strips_extras(self, tmp_path):
        f = tmp_path / "setup.py"
        f.write_text(
            'setup(\n'
            '    install_requires=[\n'
            '        "requests[security]>=2.0",\n'
            '    ]\n'
            ')\n'
        )
        result = _parse_setup_py(f)
        assert len(result) == 1
        assert "requests" in result[0]

    def test_multiline_list(self, tmp_path):
        f = tmp_path / "setup.py"
        f.write_text(
            'setup(\n'
            '    install_requires=[\n'
            '        "requests>=2.0",\n'
            '        "flask",\n'
            '    ]\n'
            ')\n'
        )
        result = _parse_setup_py(f)
        assert "requests" in result
        assert "flask" in result

    @pytest.mark.parametrize(
        "content, expected",
        [
            ("", []),
            ('setup(install_requires=["requests>=2.0"])', ["requests"]),
            ('setup(name="mypackage", version="1.0")', []),
        ],
    )
    def test_parses(self, tmp_path, content, expected):
        f = tmp_path / "setup.py"
        f.write_text(content)
        assert _parse_setup_py(f) == expected


class TestParseGoMod:
    def test_missing_file(self, tmp_path):
        assert _parse_go_mod(tmp_path / "go.mod") == []

    def test_includes_indirect(self, tmp_path):
        f = tmp_path / "go.mod"
        f.write_text(
            "module example.com/mymod\n\nrequire (\n"
            "\tgithub.com/foo/bar v1.0.0\n"
            "\tgithub.com/baz/qux v2.0.0 // indirect\n"
            ")\n"
        )
        result = _parse_go_mod(f)
        assert "github.com/foo/bar" in result
        assert "github.com/baz/qux" in result

    @pytest.mark.parametrize(
        "content, expected",
        [
            ("", []),
            (
                "module example.com/mymod\n\nrequire (\n\tgithub.com/foo/bar v1.0.0\n)\n",
                ["github.com/foo/bar"],
            ),
            (
                "module example.com/mymod\n\nrequire github.com/foo/bar v1.0.0\n",
                ["github.com/foo/bar"],
            ),
            (
                "module example.com/mymod\n\nrequire (\n"
                "\n"
                "\t// a comment\n"
                "\tgithub.com/foo/bar v1.0.0\n"
                ")\n",
                ["github.com/foo/bar"],
            ),
        ],
    )
    def test_parses(self, tmp_path, content, expected):
        f = tmp_path / "go.mod"
        f.write_text(content)
        assert _parse_go_mod(f) == expected


class TestParsePomXml:
    def test_missing_file(self, tmp_path):
        assert _parse_pom_xml(tmp_path / "pom.xml") == []

    def test_multiple_deps(self, tmp_path):
        f = tmp_path / "pom.xml"
        f.write_text(
            "<project><dependencies>"
            "<dependency><groupId>org.springframework</groupId><artifactId>spring-core</artifactId></dependency>"
            "<dependency><groupId>junit</groupId><artifactId>junit</artifactId></dependency>"
            "</dependencies></project>"
        )
        result = _parse_pom_xml(f)
        assert "org.springframework:spring-core" in result
        assert "junit:junit" in result

    @pytest.mark.parametrize(
        "content, expected",
        [
            ("", []),
            (
                "<project><dependencies>"
                "<dependency>"
                "<groupId>org.springframework</groupId>"
                "<artifactId>spring-core</artifactId>"
                "</dependency>"
                "</dependencies></project>",
                ["org.springframework:spring-core"],
            ),
            (
                '<project xmlns="http://maven.apache.org/POM/4.0.0">'
                "<dependencies>"
                "<dependency>"
                "<groupId>com.fasterxml.jackson.core</groupId>"
                "<artifactId>jackson-databind</artifactId>"
                "</dependency>"
                "</dependencies></project>",
                ["com.fasterxml.jackson.core:jackson-databind"],
            ),
            ("<project><dependencies>", []),
        ],
    )
    def test_parses(self, tmp_path, content, expected):
        f = tmp_path / "pom.xml"
        f.write_text(content)
        assert _parse_pom_xml(f) == expected


class TestParseBuildGradle:
    def test_missing_file(self, tmp_path):
        assert _parse_build_gradle(tmp_path / "build.gradle") == []

    @pytest.mark.parametrize(
        "content, expected",
        [
            ("", []),
            (
                "dependencies {\n    implementation 'com.google.guava:guava:31.0'\n}\n",
                ["com.google.guava:guava:31.0"],
            ),
            (
                'dependencies {\n    api "org.springframework:spring-core:5.0"\n}\n',
                ["org.springframework:spring-core:5.0"],
            ),
            (
                "dependencies {\n    compile 'log4j:log4j:1.2'\n}\n",
                ["log4j:log4j:1.2"],
            ),
            (
                "plugins {\n    id 'java'\n}\napply plugin: 'application'\n"
                "dependencies {\n    implementation 'com.google.guava:guava:31.0'\n}\n",
                ["com.google.guava:guava:31.0"],
            ),
        ],
    )
    def test_parses(self, tmp_path, content, expected):
        f = tmp_path / "build.gradle"
        f.write_text(content)
        assert _parse_build_gradle(f) == expected
