# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from nemo_run.core.execution.utils import (
    RFC1123_LABEL_MAX_LENGTH,
    fill_template,
    sanitize_k8s_name,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        # Already-valid names are unchanged.
        ("nemotron3-30b-bf16", "nemotron3-30b-bf16"),
        # The legacy idiom only handled '_' and '.'.
        ("Model_30B__FP8.", "model-30b-fp8"),
        ("UPPER_Case", "upper-case"),
        ("a..b__c  d", "a-b-c-d"),
        # The core fix: strict webhooks (Run:ai) reject leading/trailing '-'.
        ("_exp.1", "exp-1"),
        ("trailing-", "trailing"),
        ("-leading", "leading"),
        # All-punctuation falls back rather than producing an empty name.
        ("---", "job"),
    ],
)
def test_sanitize_k8s_name_rfc1123(raw, expected):
    assert sanitize_k8s_name(raw) == expected


def test_sanitize_k8s_name_truncates_to_label_limit():
    assert len(sanitize_k8s_name("x" * 70)) == RFC1123_LABEL_MAX_LENGTH


def test_sanitize_k8s_name_strips_hyphen_exposed_by_truncation():
    # Truncation lands on a '-'; the final strip must remove it so the name still
    # ends in an alphanumeric (RFC 1123).
    assert sanitize_k8s_name("a" * 62 + "-bbb") == "a" * 62


def test_sanitize_k8s_name_respects_custom_max_length():
    assert sanitize_k8s_name("abcdefghij", max_length=4) == "abcd"


def test_sanitize_k8s_name_rejects_empty():
    with pytest.raises(ValueError):
        sanitize_k8s_name("")


def test_fill_template_file_not_found():
    template_name = "non_existing_template.j2"
    variables = {"var1": "value1"}

    with pytest.raises(FileNotFoundError):
        fill_template(template_name, variables)


def test_fill_template_invalid_extension():
    template_name = "invalid_extension.txt"
    variables = {"var1": "value1"}

    with pytest.raises(AssertionError):
        fill_template(template_name, variables)
