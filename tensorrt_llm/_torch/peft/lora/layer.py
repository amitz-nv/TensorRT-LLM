from enum import IntEnum
from typing import Dict, List, Optional

import torch


class LoraModuleType(IntEnum):
    """Enum class representing different types of modules that can have LoRA adapters.

    This enum maps to the different attention and MLP components in a transformer model
    that can be adapted using LoRA weights.
    """
    ATTENTION_QKV = 0  # Combined QKV projection
    ATTENTION_Q = 1  # Query projection
    ATTENTION_K = 2  # Key projection
    ATTENTION_V = 3  # Value projection
    ATTENTION_DENSE = 4  # Output projection after attention

    MLP_H_TO_4H = 5  # First MLP projection (hidden to 4x hidden)
    MLP_4H_TO_H = 6  # Second MLP projection (4x hidden back to hidden)
    MLP_GATE = 7  # Gate projection in MLP

    CROSS_ATTENTION_QKV = 8  # Cross-attention QKV projection
    CROSS_ATTENTION_Q = 9  # Cross-attention Query projection
    CROSS_ATTENTION_K = 10  # Cross-attention Key projection
    CROSS_ATTENTION_V = 11  # Cross-attention Value projection
    CROSS_ATTENTION_DENSE = 12  # Cross-attention output projection

    MOE_H_TO_4H = 13  # MoE first projection
    MOE_4H_TO_H = 14  # MoE second projection
    MOE_GATE = 15  # MoE gate projection
    MOE_ROUTER = 16  # MoE router

    MLP_ROUTER = 17  # MLP router
    MLP_GATE_UP = 18  # Combined gate and up projections

    def __str__(self):
        """Return the name of the enum value."""
        return self.name

    @classmethod
    def from_string(cls, name: str) -> "LoraModuleType":
        """Convert a string to the corresponding LoraModuleType.

        Args:
            name: The string name of the module type

        Returns:
            The corresponding LoraModuleType enum value

        Raises:
            ValueError: If the name doesn't match any LoraModuleType
        """
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"Unknown LoRA module type: {name}")

    @property
    def is_attention(self) -> bool:
        """Check if this is an attention module type."""
        return self in {
            self.ATTENTION_QKV, self.ATTENTION_Q, self.ATTENTION_K,
            self.ATTENTION_V, self.ATTENTION_DENSE, self.CROSS_ATTENTION_QKV,
            self.CROSS_ATTENTION_Q, self.CROSS_ATTENTION_K,
            self.CROSS_ATTENTION_V, self.CROSS_ATTENTION_DENSE
        }

    @property
    def is_mlp(self) -> bool:
        """Check if this is an MLP module type."""
        return self in {
            self.MLP_H_TO_4H, self.MLP_4H_TO_H, self.MLP_GATE, self.MLP_GATE_UP,
            self.MLP_ROUTER
        }

    @property
    def is_moe(self) -> bool:
        """Check if this is a Mixture of Experts (MoE) module type."""
        return self in {
            self.MOE_H_TO_4H, self.MOE_4H_TO_H, self.MOE_GATE, self.MOE_ROUTER
        }


class LoraLayer(torch.nn.Module):

    def __init__(self, lora_module_types: List[LoraModuleType],
                 output_hidden_sizes: List[int],
                 layer_idx: int):
        super().__init__()

        self.lora_module_types = lora_module_types
        self.output_hidden_sizes = output_hidden_sizes
        self.layer_idx = layer_idx
        assert len(lora_module_types) == len(output_hidden_sizes)

        """
        if self.output_hidden_sizes:
            #print(f"ZUKER - LoraLayer.__init__ - {self.output_hidden_sizes=}")
            # TODO: Does device="cuda" work with TP>1?
            self._workspace_tensor = torch.zeros(33560000,
                                                 dtype=torch.uint8,
                                                 device="cuda")

            self._max_num_tokens = 256  # TODO: Get real value from somewhere
            self._output_tensors = [
                torch.zeros((self._max_num_tokens, self.output_hidden_sizes[i]),
                            dtype=torch.float16,
                            device="cuda")
                for i in range(len(self.output_hidden_sizes))
            ]
            self._output_tensors_ptrs = [
                t.data_ptr() for t in self._output_tensors
            ]
        """

    def get_lora_ranks_weight_pointers_and_active_lora_module_ids(self, lora_params: Dict) -> tuple[list[int], list[int], list[int]]:
        lora_ranks = []
        lora_weight_pointers = []
        active_lora_module_ids = []
        for module_idx in self.lora_module_types:
            module_idx = int(module_idx)
            if module_idx in lora_params[self.layer_idx]:
                active_lora_module_ids.append(module_idx)
                lora_ranks.append(
                    lora_params[self.layer_idx][module_idx]['adapter_size'])
                lora_weight_pointers.append(
                    lora_params[self.layer_idx][module_idx]['weight_pointers'])
        return lora_ranks, lora_weight_pointers, active_lora_module_ids

    def get_lora_ranks_weight_pointers_of_lora_module_types(self, lora_params: Dict, zero_adapter_size: int, zero_weight_in_ptr: int, zero_weight_out_ptr: int) -> tuple[list[int], list[int], list[int]]:
        lora_ranks = []
        lora_weight_pointers = []
        active_lora_module_ids = []
        for module_idx in self.lora_module_types:
            module_idx = int(module_idx)
            if module_idx in lora_params[self.layer_idx]:
                active_lora_module_ids.append(module_idx)
                lora_ranks.append(
                    lora_params[self.layer_idx][module_idx]['adapter_size'])
                lora_weight_pointers.append(
                    lora_params[self.layer_idx][module_idx]['weight_pointers'])
            else:
                lora_ranks.append(torch.IntTensor([zero_adapter_size]))
                lora_weight_pointers.append(torch.LongTensor([zero_weight_in_ptr, zero_weight_out_ptr, 0]))
        return lora_ranks, lora_weight_pointers, active_lora_module_ids

    def forward(
        self,
        x,
        lora_params: Dict,
    ) -> Optional[torch.Tensor]:
        print("ZUKER - LoraLayer.forward - start")
        if bool(lora_params):
            #print(f"ZUKER - LoraLayer.forward - {x.dtype=}, {self.layer_idx=}, {lora_params[self.layer_idx]=}")
            lora_ranks, lora_weight_pointers, active_lora_module_ids = \
                self.get_lora_ranks_weight_pointers_and_active_lora_module_ids(lora_params)

            # Values are present only when CUDA graphs are enabled
            cuda_graphs_metadata = lora_params[self.layer_idx].get(tuple(self.output_hidden_sizes), {})
            output_tensors = cuda_graphs_metadata.get('output_tensors', None)
            output_tensors_ptrs = cuda_graphs_metadata.get('output_tensors_ptrs', None)
            workspace_tensor_device_group_gemm1 = cuda_graphs_metadata.get('workspace_tensor_device_splitk_group_gemm', None)
            workspace_tensor_host_group_gemm1 = cuda_graphs_metadata.get('workspace_tensor_host_splitk_group_gemm', None)
            workspace_tensor_device_group_gemm2: torch.Tensor = cuda_graphs_metadata.get('workspace_tensor_device_group_gemm', None)
            workspace_tensor_host_group_gemm2 = cuda_graphs_metadata.get('workspace_tensor_host_group_gemm', None)

            self.input_tensor = x
            print(f"ZUKER - LoraLayer.forward - {self.layer_idx=}, {self.output_hidden_sizes=}, {hex(x.data_ptr())=}")

            num_seqs = lora_params['num_seqs']

            if len(active_lora_module_ids) == 0:
                print("ZUKER - LoraLayer.forward - no active_lora_module_ids, returning None")
                return None
            else:
                print(f"ZUKER - LoraLayer.forward - {num_seqs=}, {self.lora_module_types=}, {x.shape=}")
                print("ZUKER - LoraLayer.forward - calling lora_grouped_gemm")
                lora_outputs = torch.ops.trtllm.lora_grouped_gemm(
                    x,
                    lora_params['host_request_types'][:num_seqs],
                    lora_ranks,
                    lora_weight_pointers,
                    lora_params['prompt_lens_cpu'][:num_seqs],
                    self.output_hidden_sizes,
                    False,  # transA
                    True,  # transB
                    max([r.max() for r in lora_ranks]),
                    0,
                    output_tensors is None,  # TODO smor- should be lora_params["remove_input_padding"], support in loraOp as well
                    output_tensors_ptrs,
                    workspace_tensor_device_group_gemm1,
                    workspace_tensor_host_group_gemm1,
                    workspace_tensor_device_group_gemm2,
                    workspace_tensor_host_group_gemm2,
                )
                #print(f"ZUKER - LoraLayer.forward - {self.layer_idx=}, {output_tensors=}, {workspace_tensor=}")
                #print(f"ZUKER - LoraLayer.forward - {lora_params[self.layer_idx].keys()=}")
                if output_tensors is not None:
                    # working version:
                    lora_outputs = output_tensors[:len(self.output_hidden_sizes)]
                    print(f"ZUKER - LoraLayer.forward - lora_outputs shapes - {[t.shape for t in lora_outputs]=}")
                    return torch.cat(lora_outputs, dim=-1)[:x.shape[0]]
                    # test version:
                    #lora_outputs = output_tensors[:len(active_lora_module_ids)]
                #print(f"ZUKER - LoraLayer.forward - {self.output_hidden_sizes=}")
                #print(f"ZUKER - LoraLayer.forward - {[hex(a) for a in self._output_tensors_ptrs]}")
                #print(f"ZUKER - LoraLayer.forward - {lora_outputs=}")
                #if isinstance(lora_outputs, torch.Tensor):
                #    return lora_outputs
                #else:
                # For multiple LoRA modules, some might not be executed in grouped gemm.
                # For those modules not executed, we create zero tensors with matching dimensions.
                # Finally we concatenate all tensors (both LoRA outputs and zero tensors) in order.
                # NOTE: For example: /home/scratch.trt_llm_data/llm-models/llama-models/luotuo-lora-7b-0.1 doesn't have
                #       anything for ATTENTION_K
                lora_output = []
                idx = 0
                #if isinstance(lora_outputs, torch.Tensor):
                #    print(f"ZUKER - LoraLayer.forward - {lora_outputs.shape=}")
                #else:
                #    print(f"ZUKER - LoraLayer.forward - {len(lora_outputs)=}")
                for module_idx in self.lora_module_types:
                    if int(module_idx) in active_lora_module_ids:
                        print(f"ZUKER - LoraLayer.forward - Adding lora output for module {module_idx}, {idx=}")
                        lora_output.append(lora_outputs[idx][:x.shape[0]])
                        idx += 1
                    else:
                        print(f"ZUKER - LoraLayer.forward - Adding zeros as lora output for module {module_idx}, {idx=}")
                        lora_output.append(
                            torch.zeros(list(x.shape[:-1]) + [
                                self.output_hidden_sizes[
                                    self.lora_module_types.index(module_idx)]
                            ],
                                        dtype=x.dtype,
                                        device=x.device))
                print(f"ZUKER - LoraLayer.forward - {[w.shape for w in lora_output]=}")
                lora_output = torch.cat(lora_output, dim=-1)
                return lora_output

        else:
            return None
