"""nsys stats CSV summaries (C4 task 6). Headers as nsys 2026.5.1 writes them."""

import pytest

from edgeinfer.perf.nsys import NsysFormatError, summarize_table

KERN = """Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
60.0,3000000,6,500000.0,500000.0,1,1,0.0,gemm_a
40.0,2000000,12,166666.7,1.0,1,1,0.0,"conv_b, with comma"
"""
MEM = """Time (%),Total Time (ns),Count,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Operation
80.0,4000000,4,1.0,1.0,1,1,0.0,[CUDA memcpy Host-to-Device]
20.0,1000000,4,1.0,1.0,1,1,0.0,[CUDA memcpy Device-to-Host]
"""


def test_kernel_summary_ranks_by_total_time():
    s = summarize_table(KERN, "cuda_gpu_kern_sum", top=1)
    assert s["total_ms"] == pytest.approx(5.0) and s["entries"] == 2
    assert s["top"] == [{"name": "gemm_a", "total_ms": 3.0, "count": 6, "share": 0.6}]


def test_memcpy_summary_uses_its_own_columns():
    s = summarize_table(MEM, "cuda_gpu_mem_time_sum")
    assert [t["name"] for t in s["top"]] == ["[CUDA memcpy Host-to-Device]",
                                             "[CUDA memcpy Device-to-Host]"]  # fmt: skip


def test_unexpected_header_is_a_specific_error():
    with pytest.raises(NsysFormatError, match="lacks columns"):
        summarize_table(MEM, "cuda_gpu_kern_sum")
