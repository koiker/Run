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

import os
import re
from typing import Optional

import jinja2

# RFC 1123 DNS label limit, used for Kubernetes object names (Jobs, PyTorchJobs,
# TrainJobs) and the pod names schedulers derive from them.
RFC1123_LABEL_MAX_LENGTH = 63


def sanitize_k8s_name(name: str, max_length: int = RFC1123_LABEL_MAX_LENGTH) -> str:
    """Coerce an arbitrary string into a valid RFC 1123 DNS label.

    Kubernetes object names — and the pod names schedulers derive from them — must
    be a lowercase RFC 1123 DNS label: ``[a-z0-9]([-a-z0-9]*[a-z0-9])?`` with a
    maximum length of 63 characters. Several schedulers enforce this strictly and
    reject names that merely *contain* the right characters but begin or end with
    a ``-`` — notably **NVIDIA Run:ai**, where a job name derived from an
    underscore- or timestamp-prefixed experiment id (e.g. ``_exp.1`` ->
    ``-exp-1``) is refused by the admission webhook.

    The previous ``name.replace("_", "-").replace(".", "-").lower()`` idiom fixed
    only ``_`` and ``.`` and could still emit leading/trailing hyphens or other
    invalid characters. This normalizes by lowercasing, collapsing every run of
    invalid characters into a single ``-``, truncating to ``max_length``, and
    stripping leading/trailing ``-`` so the result begins and ends with an
    alphanumeric.

    Args:
        name: Arbitrary candidate name.
        max_length: Maximum label length (default 63, the RFC 1123 label limit).

    Returns:
        A valid RFC 1123 DNS label. Falls back to ``"job"`` if *name* sanitizes to
        an empty string (e.g. it was entirely punctuation).

    Raises:
        ValueError: if *name* is empty.
    """
    if not name:
        raise ValueError("name must be a non-empty string")

    # Lowercase, then collapse any run of non-[a-z0-9] characters into a single
    # hyphen (covers '_', '.', whitespace, and anything else a caller passes in).
    sanitized = re.sub(r"[^a-z0-9]+", "-", name.lower())
    # Truncate before the final strip so a hyphen exposed at the cut point is
    # also removed, guaranteeing the last character is alphanumeric.
    sanitized = sanitized[:max_length].strip("-")
    return sanitized or "job"


def fill_template(template_name: str, variables: dict, template_dir: Optional[str] = None) -> str:
    """Create a file from a Jinja template and return the filename."""
    assert template_name.endswith(".j2"), template_name
    template_dir = template_dir or os.path.join(os.path.dirname(__file__), "templates")
    template_path = os.path.join(template_dir, template_name)
    if not os.path.exists(template_path):
        raise FileNotFoundError(f'Template "{template_path}" does not exist.')
    with open(template_path, "r", encoding="utf-8") as fin:
        template = fin.read()

    j2_template = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
    ).from_string(template)
    content = j2_template.render(**variables)
    return content
