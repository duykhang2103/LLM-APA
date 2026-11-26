from llm_grading.data.evidence import evidence_code, evidence_log, uncomment_whole_file


def test_whole_file_commented_is_uncommented():
    code = "//#include <iostream>\n//int main() {\n//\treturn 0;\n//}\n"
    assert uncomment_whole_file(code) == "#include <iostream>\nint main() {\n\treturn 0;\n}"


def test_block_comment_file_is_uncommented():
    assert "int main()" in uncomment_whole_file("/*\nint main() { return 0; }\n*/")


def test_normal_or_empty_file_is_unchanged():
    code = "int main() { return 0; } // done"
    assert uncomment_whole_file(code) is None and evidence_code(code) == code
    assert uncomment_whole_file("") is None
    assert uncomment_whole_file("// TODO") is None


def test_log_mentions_commented_code():
    log = evidence_log({"strict": True, "relaxed": True, "errors": [], "commented_out": True})
    assert log.startswith("The whole file is commented out") and "compiled without errors" in log
