# Copyright 2019 The Lifetime Value Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
licenses(["notice"])

exports_files([
    "LICENSE",
])

pytype_library(
    name = "lifetime_value",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":metrics",
        ":zero_inflated_lognormal",
    ],
)

pytype_library(
    name = "metrics",
    srcs = [
        "metrics.py",
    ],
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # pandas dep,
        # sklearn dep,
        # typing dep,
    ],
)

py_test(
    name = "metrics_test",
    size = "small",
    srcs = [
        "metrics_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":metrics",
        "//testing/pybase",
        # numpy dep,
        # pandas dep,
    ],
)

pytype_library(
    name = "zero_inflated_lognormal",
    srcs = [
        "zero_inflated_lognormal.py",
    ],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)

py_test(
    name = "zero_inflated_lognormal_test",
    size = "small",
    srcs = [
        "zero_inflated_lognormal_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":zero_inflated_lognormal",
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
    ],
)
