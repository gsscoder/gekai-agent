from agent.manifest_parsers import (
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

    def test_empty_file(self, tmp_path):
        f = tmp_path / "requirements.txt"
        f.write_text("")
        assert _parse_requirements_txt(f) == []

    def test_strips_version_specifiers(self, tmp_path):
        f = tmp_path / "requirements.txt"
        f.write_text("requests>=2.0\n")
        assert _parse_requirements_txt(f) == ["requests"]

    def test_strips_extras(self, tmp_path):
        f = tmp_path / "requirements.txt"
        f.write_text("requests[security]>=2.0\n")
        assert _parse_requirements_txt(f) == ["requests"]

    def test_skips_flags(self, tmp_path):
        f = tmp_path / "requirements.txt"
        f.write_text("-r other.txt\n-c constraints.txt\n-e git+https://example.com\n")
        assert _parse_requirements_txt(f) == []

    def test_skips_comments_and_blanks(self, tmp_path):
        f = tmp_path / "requirements.txt"
        f.write_text("# comment\n\nrequests\n")
        assert _parse_requirements_txt(f) == ["requests"]

    def test_multiple_packages(self, tmp_path):
        f = tmp_path / "requirements.txt"
        f.write_text("flask\ndjango\nfastapi\n")
        assert _parse_requirements_txt(f) == ["flask", "django", "fastapi"]


class TestParsePackageJson:
    def test_missing_file(self, tmp_path):
        assert _parse_package_json(tmp_path / "package.json") == []

    def test_empty_file(self, tmp_path):
        f = tmp_path / "package.json"
        f.write_text("")
        assert _parse_package_json(f) == []

    def test_dependencies_and_dev_dependencies(self, tmp_path):
        f = tmp_path / "package.json"
        f.write_text('{"dependencies": {"react": "^18.0"}, "devDependencies": {"typescript": "^5.0"}}')
        result = _parse_package_json(f)
        assert "react" in result
        assert "typescript" in result

    def test_missing_keys(self, tmp_path):
        f = tmp_path / "package.json"
        f.write_text('{"name": "my-app", "version": "1.0.0"}')
        assert _parse_package_json(f) == []

    def test_invalid_json(self, tmp_path):
        f = tmp_path / "package.json"
        f.write_text("{not valid json")
        assert _parse_package_json(f) == []


class TestParsePyprojectToml:
    def test_missing_file(self, tmp_path):
        assert _parse_pyproject_toml(tmp_path / "pyproject.toml") == []

    def test_empty_file(self, tmp_path):
        f = tmp_path / "pyproject.toml"
        f.write_text("")
        assert _parse_pyproject_toml(f) == []

    def test_project_dependencies_section(self, tmp_path):
        f = tmp_path / "pyproject.toml"
        f.write_text("[project.dependencies]\nrequests = \">=2.0\"\nflask = \">=2.0\"\n")
        result = _parse_pyproject_toml(f)
        assert "requests" in result
        assert "flask" in result

    def test_poetry_dependencies_section(self, tmp_path):
        f = tmp_path / "pyproject.toml"
        f.write_text("[tool.poetry.dependencies]\npython = \"^3.11\"\nhttpx = \"^0.24\"\n")
        result = _parse_pyproject_toml(f)
        assert "python" in result
        assert "httpx" in result

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

    def test_empty_file(self, tmp_path):
        f = tmp_path / "project.csproj"
        f.write_text("")
        assert _parse_csproj(f) == []

    def test_package_references(self, tmp_path):
        f = tmp_path / "project.csproj"
        f.write_text(
            '<Project><ItemGroup>'
            '<PackageReference Include="Newtonsoft.Json" Version="13.0" />'
            '</ItemGroup></Project>'
        )
        assert _parse_csproj(f) == ["Newtonsoft.Json"]

    def test_namespaced_tags(self, tmp_path):
        f = tmp_path / "project.csproj"
        f.write_text(
            '<Project xmlns="http://schemas.microsoft.com/developer/msbuild/2003">'
            '<ItemGroup>'
            '<PackageReference Include="Serilog" Version="3.0" />'
            '</ItemGroup></Project>'
        )
        assert _parse_csproj(f) == ["Serilog"]

    def test_no_packages(self, tmp_path):
        f = tmp_path / "project.csproj"
        f.write_text('<Project><ItemGroup></ItemGroup></Project>')
        assert _parse_csproj(f) == []

    def test_invalid_xml(self, tmp_path):
        f = tmp_path / "project.csproj"
        f.write_text("<Project><ItemGroup>")
        assert _parse_csproj(f) == []


class TestParseSetupPy:
    def test_missing_file(self, tmp_path):
        assert _parse_setup_py(tmp_path / "setup.py") == []

    def test_empty_file(self, tmp_path):
        f = tmp_path / "setup.py"
        f.write_text("")
        assert _parse_setup_py(f) == []

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

    def test_strips_version_specifiers(self, tmp_path):
        f = tmp_path / "setup.py"
        f.write_text('setup(install_requires=["requests>=2.0"])')
        assert _parse_setup_py(f) == ["requests"]

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

    def test_no_install_requires(self, tmp_path):
        f = tmp_path / "setup.py"
        f.write_text('setup(name="mypackage", version="1.0")')
        assert _parse_setup_py(f) == []


class TestParseGoMod:
    def test_missing_file(self, tmp_path):
        assert _parse_go_mod(tmp_path / "go.mod") == []

    def test_empty_file(self, tmp_path):
        f = tmp_path / "go.mod"
        f.write_text("")
        assert _parse_go_mod(f) == []

    def test_require_block(self, tmp_path):
        f = tmp_path / "go.mod"
        f.write_text("module example.com/mymod\n\nrequire (\n\tgithub.com/foo/bar v1.0.0\n)\n")
        assert _parse_go_mod(f) == ["github.com/foo/bar"]

    def test_single_line_require(self, tmp_path):
        f = tmp_path / "go.mod"
        f.write_text("module example.com/mymod\n\nrequire github.com/foo/bar v1.0.0\n")
        assert _parse_go_mod(f) == ["github.com/foo/bar"]

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

    def test_skips_blank_and_comment_lines(self, tmp_path):
        f = tmp_path / "go.mod"
        f.write_text(
            "module example.com/mymod\n\nrequire (\n"
            "\n"
            "\t// a comment\n"
            "\tgithub.com/foo/bar v1.0.0\n"
            ")\n"
        )
        result = _parse_go_mod(f)
        assert result == ["github.com/foo/bar"]


class TestParsePomXml:
    def test_missing_file(self, tmp_path):
        assert _parse_pom_xml(tmp_path / "pom.xml") == []

    def test_empty_file(self, tmp_path):
        f = tmp_path / "pom.xml"
        f.write_text("")
        assert _parse_pom_xml(f) == []

    def test_dependency_format(self, tmp_path):
        f = tmp_path / "pom.xml"
        f.write_text(
            "<project><dependencies>"
            "<dependency>"
            "<groupId>org.springframework</groupId>"
            "<artifactId>spring-core</artifactId>"
            "</dependency>"
            "</dependencies></project>"
        )
        assert _parse_pom_xml(f) == ["org.springframework:spring-core"]

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

    def test_namespaced_tags(self, tmp_path):
        f = tmp_path / "pom.xml"
        f.write_text(
            '<project xmlns="http://maven.apache.org/POM/4.0.0">'
            "<dependencies>"
            "<dependency>"
            "<groupId>com.fasterxml.jackson.core</groupId>"
            "<artifactId>jackson-databind</artifactId>"
            "</dependency>"
            "</dependencies></project>"
        )
        assert _parse_pom_xml(f) == ["com.fasterxml.jackson.core:jackson-databind"]

    def test_invalid_xml(self, tmp_path):
        f = tmp_path / "pom.xml"
        f.write_text("<project><dependencies>")
        assert _parse_pom_xml(f) == []


class TestParseBuildGradle:
    def test_missing_file(self, tmp_path):
        assert _parse_build_gradle(tmp_path / "build.gradle") == []

    def test_empty_file(self, tmp_path):
        f = tmp_path / "build.gradle"
        f.write_text("")
        assert _parse_build_gradle(f) == []

    def test_implementation(self, tmp_path):
        f = tmp_path / "build.gradle"
        f.write_text("dependencies {\n    implementation 'com.google.guava:guava:31.0'\n}\n")
        assert _parse_build_gradle(f) == ["com.google.guava:guava:31.0"]

    def test_api(self, tmp_path):
        f = tmp_path / "build.gradle"
        f.write_text('dependencies {\n    api "org.springframework:spring-core:5.0"\n}\n')
        assert _parse_build_gradle(f) == ["org.springframework:spring-core:5.0"]

    def test_compile(self, tmp_path):
        f = tmp_path / "build.gradle"
        f.write_text("dependencies {\n    compile 'log4j:log4j:1.2'\n}\n")
        assert _parse_build_gradle(f) == ["log4j:log4j:1.2"]

    def test_non_dep_lines_ignored(self, tmp_path):
        f = tmp_path / "build.gradle"
        f.write_text(
            "plugins {\n    id 'java'\n}\napply plugin: 'application'\n"
            "dependencies {\n    implementation 'com.google.guava:guava:31.0'\n}\n"
        )
        result = _parse_build_gradle(f)
        assert result == ["com.google.guava:guava:31.0"]
