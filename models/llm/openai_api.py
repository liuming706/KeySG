"""OpenAI GPT API interface for text, vision, and structured outputs.

Prefers the OpenAI Responses API, but falls back to Chat Completions for
OpenAI-compatible backends that do not implement ``/v1/responses``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Iterator, List, Optional, Type, Union

import numpy as np
import openai
from PIL import Image
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
from loguru import logger
from models.llm._common import encode_image_data_url

load_dotenv()

import os


class GPTInterface:
    """Interface for OpenAI GPT API supporting text, vision, structured outputs, and embeddings."""

    def __init__(
        self, client: Optional[openai.OpenAI] = None, qa_method: str = "responses_parse"
    ):
        self.qa_method = qa_method
        if client is not None:
            self.client = client
        else:
            # 从环境变量读取配置
            api_key = os.getenv("OPENAI_API_KEY", "dummy")
            base_url = os.getenv("OPENAI_BASE_URL", None)
            timeout = int(os.getenv("OPENAI_TIMEOUT", "120"))
            proxy = os.getenv("OPENAI_PROXY", None)

            # 构建 httpx 客户端
            http_client_kwargs = {"timeout": timeout}
            if proxy:
                http_client_kwargs["proxy"] = proxy

            custom_client = httpx.Client(**http_client_kwargs)

            # 构建 OpenAI 客户端
            openai_kwargs = {
                "api_key": api_key,
                "http_client": custom_client,
            }
            if base_url:
                openai_kwargs["base_url"] = base_url

            self.client = openai.OpenAI(**openai_kwargs)

    def _encode_image(
        self, image: Union[np.ndarray, Image.Image], format: str = "jpeg"
    ) -> str:
        """Encode image to data URL."""
        return encode_image_data_url(image, format=format)

    def _prepare_common_kwargs(
        self,
        instructions: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare common API arguments."""
        api_kwargs = {}
        if instructions:
            api_kwargs["instructions"] = instructions
        if reasoning_effort:
            api_kwargs["reasoning"] = {"effort": reasoning_effort}
        if verbosity:
            api_kwargs["text"] = {"verbosity": verbosity}
        api_kwargs.update(kwargs)
        return api_kwargs

    def _should_fallback_to_chat(self, exc: Exception) -> bool:
        """Return True when the backend likely lacks Responses API support."""
        msg = str(exc).lower()
        fallback_markers = (
            "404",
            "not found",
            "page not found",
            "/v1/responses",
            "unknown url",
            "unsupported",
        )
        return any(marker in msg for marker in fallback_markers)

    def _is_parsed_result_empty(self, parsed: BaseModel) -> bool:
        """Check if a parsed Pydantic model has only empty/default values.

        This detects cases where the API returns a valid type but with no real content,
        which happens with models that don't properly support structured output.
        """
        if parsed is None:
            return True
        data = parsed.model_dump()
        for key, value in data.items():
            if isinstance(value, str) and value.strip():
                return False
            if isinstance(value, list) and len(value) > 0:
                return False
            if isinstance(value, dict) and value:
                return False
            if isinstance(value, (int, float)) and value != 0:
                return False
            if isinstance(value, bool) and value:
                return False
        return True

    def _build_chat_messages(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        *,
        instructions: Optional[str] = None,
        image: Optional[Union[str, np.ndarray, Image.Image, List[Any]]] = None,
        detail: str = "auto",
    ) -> List[Dict[str, Any]]:
        """Convert prompt + optional images into Chat Completions message format."""
        messages: List[Dict[str, Any]] = []
        if instructions:
            messages.append({"role": "system", "content": instructions})

        if isinstance(prompt, str):
            if image is None:
                messages.append({"role": "user", "content": prompt})
            else:
                images = image if isinstance(image, list) else [image]
                content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
                for img in images:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    img
                                    if isinstance(img, str)
                                    else self._encode_image(img)
                                ),
                                "detail": detail,
                            },
                        }
                    )
                messages.append({"role": "user", "content": content})
            return messages

        for message in prompt:
            role = message.get("role", "user")
            content = message.get("content", "")
            if isinstance(content, list):
                converted: List[Dict[str, Any]] = []
                for item in content:
                    if not isinstance(item, dict):
                        converted.append({"type": "text", "text": str(item)})
                        continue
                    item_type = item.get("type")
                    if item_type == "input_text":
                        converted.append({"type": "text", "text": item.get("text", "")})
                    elif item_type == "input_image":
                        image_url = item.get("image_url")
                        if isinstance(image_url, dict):
                            url = image_url.get("url", "")
                        else:
                            url = image_url
                        converted.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": url,
                                    "detail": item.get("detail", detail),
                                },
                            }
                        )
                    else:
                        converted.append(item)
                messages.append({"role": role, "content": converted})
            else:
                messages.append({"role": role, "content": content})
        return messages

    def _extract_chat_text(self, response: Any) -> str:
        """Extract text from a Chat Completions response."""
        choice = response.choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(text)
                else:
                    text = getattr(item, "text", None)
                    if text:
                        parts.append(text)
            return "\n".join(parts)
        return str(content)

    def _chat_completion_create(self, **kwargs: Any) -> Any:
        return self.client.chat.completions.create(**kwargs)

    def _strip_markdown_code_blocks(self, text: str) -> str:
        """Strip markdown code block fences from LLM output.

        Many models (especially non-OpenAI models) wrap JSON in markdown
        code blocks like ```json ... ``` or ``` ... ```. This method
        strips those fences to extract the raw JSON.
        """
        import re as _re

        # Match ```json\n...\n``` or ```\n...\n```
        # Use non-greedy match to handle multiple code blocks
        pattern = r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```"
        match = _re.search(pattern, text, _re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _structured_via_chat(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        response_model: Type[BaseModel],
        model: str,
        image: Optional[Union[str, np.ndarray, Image.Image, List[Any]]] = None,
        detail: str = "auto",
        instructions: Optional[str] = None,
        **_: Any,
    ) -> BaseModel:
        """Fallback structured output generation via Chat Completions."""
        schema = json.dumps(
            response_model.model_json_schema(), ensure_ascii=False, indent=2
        )
        format_hint = (
            "Return ONLY valid JSON matching this schema exactly. "
            "Do not wrap it in markdown fences.\n\n"
            f"JSON schema:\n{schema}"
        )
        merged_instructions = (
            f"{instructions}\n\n{format_hint}" if instructions else format_hint
        )
        messages = self._build_chat_messages(
            prompt,
            instructions=merged_instructions,
            image=image,
            detail=detail,
        )
        response = self._chat_completion_create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        text = self._extract_chat_text(response)

        # Log raw response for debugging
        logger.debug(
            "[_structured_via_chat] model={}, raw_response_length={}, raw_response_preview={}",
            model,
            len(text),
            text[:500] if text else "(empty)",
        )

        # First, strip markdown code blocks if present
        cleaned_text = self._strip_markdown_code_blocks(text)

        try:
            return response_model.model_validate_json(cleaned_text)
        except Exception as e:
            logger.warning(
                "[_structured_via_chat] Direct parse failed: {}, trying fallback extraction",
                str(e)[:200],
            )
            # LLM returned non-JSON text; try to extract JSON from it
            import re as _re

            # Try to extract the "properties" wrapper that some LLMs (e.g., qwen) produce.
            # The LLM may return {"properties": {...}} instead of the flat object.
            try:
                raw_obj = json.loads(cleaned_text)
                if isinstance(raw_obj, dict) and "properties" in raw_obj:
                    props = raw_obj["properties"]
                    if isinstance(props, dict):
                        logger.info(
                            "[_structured_via_chat] Extracted 'properties' wrapper for model {}",
                            response_model.__name__,
                        )
                        return response_model.model_validate(props)
            except Exception:
                pass  # Fall through to other extraction methods

            # First, try to extract a JSON array [...] and wrap it into the model's List field
            array_match = _re.search(r"\[.*\]", cleaned_text, _re.DOTALL)
            if array_match:
                try:
                    parsed_list = json.loads(array_match.group())
                    if isinstance(parsed_list, list):
                        # Find the first List[str] field in the model and wrap the list
                        for (
                            field_name,
                            field_info,
                        ) in response_model.model_fields.items():
                            annotation = field_info.annotation
                            origin = getattr(annotation, "__origin__", None)
                            if origin is list:
                                logger.info(
                                    "[_structured_via_chat] Extracted list with {} items for model {}",
                                    len(parsed_list),
                                    response_model.__name__,
                                )
                                return response_model(**{field_name: parsed_list})
                except Exception as list_e:
                    logger.warning(
                        "[_structured_via_chat] Array extraction failed: {}", list_e
                    )

            # Then, try to extract a JSON object {...}
            json_match = _re.search(r"\{.*\}", cleaned_text, _re.DOTALL)
            if json_match:
                try:
                    result = response_model.model_validate_json(json_match.group())
                    logger.info(
                        "[_structured_via_chat] Extracted JSON object for model {}",
                        response_model.__name__,
                    )
                    return result
                except Exception as obj_e:
                    logger.warning(
                        "[_structured_via_chat] JSON object extraction failed: {}, matched_text={}",
                        obj_e,
                        json_match.group()[:200],
                    )

            # Last resort: wrap the raw text into the first string field of the model
            field_name = next(
                (
                    f
                    for f, info in response_model.model_fields.items()
                    if info.annotation is str
                ),
                None,
            )
            if field_name:
                logger.warning(
                    "[_structured_via_chat] All JSON extraction failed, wrapping raw text in field '{}'",
                    field_name,
                )
                return response_model(**{field_name: text})

            logger.error(
                "[_structured_via_chat] Complete failure for model {}, response_model={}, text_preview={}",
                model,
                response_model.__name__,
                text[:300] if text else "(empty)",
            )
            raise

    def text_prompt(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        model: str = "gpt-5.4",
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[str, Iterator[str]]:
        """Generate text response from a prompt."""
        api_kwargs = self._prepare_common_kwargs(**kwargs)
        api_kwargs.update({"model": model, "input": prompt})
        instructions = api_kwargs.pop("instructions", None)

        try:
            if stream:
                response_stream = self.client.responses.create(
                    stream=True, **api_kwargs
                )
                return (
                    chunk.delta
                    for chunk in response_stream
                    if chunk.type == "response.output_text.delta"
                )

            response = self.client.responses.create(**api_kwargs)
            return response.output_text
        except Exception as e:
            if not self._should_fallback_to_chat(e):
                raise
            messages = self._build_chat_messages(prompt, instructions=instructions)
            response = self._chat_completion_create(model=model, messages=messages)
            return self._extract_chat_text(response)

    def vision_prompt(
        self,
        prompt: str,
        image: Union[
            str, np.ndarray, Image.Image, List[Union[str, np.ndarray, Image.Image]]
        ],
        model: str = "gpt-5.4",
        detail: str = "auto",
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[str, Iterator[str]]:
        """Analyze image(s) and generate text response."""
        api_kwargs = self._prepare_common_kwargs(**kwargs)
        instructions = api_kwargs.get("instructions")

        images = image if isinstance(image, list) else [image]
        image_contents = [
            {
                "type": "input_image",
                "image_url": img if isinstance(img, str) else self._encode_image(img),
                "detail": detail,
            }
            for img in images
        ]

        content = [{"type": "input_text", "text": prompt}, *image_contents]
        api_kwargs.update(
            {"model": model, "input": [{"role": "user", "content": content}]}
        )

        try:
            if stream:
                response_stream = self.client.responses.create(
                    stream=True, **api_kwargs
                )
                return (
                    chunk.delta
                    for chunk in response_stream
                    if chunk.type == "response.output_text.delta"
                )

            response = self.client.responses.create(**api_kwargs)
            return response.output_text
        except Exception as e:
            if not self._should_fallback_to_chat(e):
                raise
            messages = self._build_chat_messages(
                prompt,
                instructions=instructions,
                image=image,
                detail=detail,
            )
            response = self._chat_completion_create(model=model, messages=messages)
            return self._extract_chat_text(response)

    def structured_prompt(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        response_model: Type[BaseModel],
        model: str = "gpt-5.4",
        stream: bool = False,
        image: Optional[Union[str, np.ndarray, Image.Image, List]] = None,
        detail: str = "auto",
        **kwargs: Any,
    ) -> Union[BaseModel, Iterator[Any]]:
        """Generate structured JSON response conforming to a Pydantic model.

        When ``qa_method`` is ``"chat"``, uses Chat Completions API directly.
        When ``qa_method`` is ``"responses_parse"`` (default), uses OpenAI Responses API
        with fallback to Chat Completions.
        """
        # Extract per-call method override (if any)
        method = kwargs.pop("method", None)
        effective_method = method or self.qa_method

        # If effective_method is "chat", use Chat Completions directly
        if effective_method == "chat":
            instructions = kwargs.get("instructions")
            logger.debug("[structured_prompt] Using chat method, model={}", model)
            return self._structured_via_chat(
                prompt,
                response_model=response_model,
                model=model,
                image=image,
                detail=detail,
                instructions=instructions,
            )

        api_kwargs = self._prepare_common_kwargs(**kwargs)
        instructions = api_kwargs.get("instructions")

        final_input: Union[str, List[Dict[str, Any]]]
        if image:
            images = image if isinstance(image, list) else [image]
            image_contents = [
                {
                    "type": "input_image",
                    "image_url": (
                        img if isinstance(img, str) else self._encode_image(img)
                    ),
                    "detail": detail,
                }
                for img in images
            ]

            if isinstance(prompt, str):
                content = [{"type": "input_text", "text": prompt}, *image_contents]
                final_input = [{"role": "user", "content": content}]
            else:
                final_input = [*prompt, {"role": "user", "content": image_contents}]
        else:
            final_input = prompt

        api_kwargs.update(
            {
                "model": model,
                "input": final_input,
                "text_format": response_model,
            }
        )

        try:
            if stream:
                return self.client.responses.stream(**api_kwargs)

            response = self.client.responses.parse(**api_kwargs)
            parsed = response.output_parsed
            # If output_parsed is not the expected type or is empty, fall back
            if not isinstance(parsed, response_model) or self._is_parsed_result_empty(
                parsed
            ):
                logger.info(
                    "[structured_prompt] parsed empty/wrong type (type={}, empty={}), "
                    "falling back to _structured_via_chat, model={}",
                    type(parsed).__name__,
                    self._is_parsed_result_empty(parsed) if parsed else True,
                    model,
                )
                return self._structured_via_chat(
                    prompt,
                    response_model=response_model,
                    model=model,
                    image=image,
                    detail=detail,
                    instructions=instructions,
                )
            return parsed
        except Exception as e:
            logger.warning(
                "[structured_prompt] responses.parse failed ({}), "
                "falling back to _structured_via_chat, model={}",
                e,
                model,
            )
            return self._structured_via_chat(
                prompt,
                response_model=response_model,
                model=model,
                image=image,
                detail=detail,
                instructions=instructions,
            )

    async def _process_one_structured_prompt(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        response_model: Type[BaseModel],
        model: str,
        image: Optional[Union[str, np.ndarray, Image.Image, List]],
        detail: str,
        **kwargs: Any,
    ) -> BaseModel:
        """Helper for async structured prompt processing.

        When ``qa_method`` is ``"chat"``, uses Chat Completions API directly.
        When ``qa_method`` is ``"responses_parse"`` (default), uses OpenAI Responses API
        with fallback to Chat Completions.
        """
        # Extract per-call method override (if any)
        method = kwargs.pop("method", None)
        effective_method = method or self.qa_method

        # If effective_method is "chat", use Chat Completions directly
        if effective_method == "chat":
            instructions = kwargs.get("instructions")
            logger.debug(
                "[_process_one_structured_prompt] Using chat method, model={}", model
            )
            return await asyncio.to_thread(
                self._structured_via_chat,
                prompt,
                response_model,
                model,
                image,
                detail,
                instructions,
            )

        api_kwargs = self._prepare_common_kwargs(**kwargs)

        final_input: Union[str, List[Dict[str, Any]]]
        if image:
            images = image if isinstance(image, list) else [image]
            image_contents = [
                {
                    "type": "input_image",
                    "image_url": (
                        img if isinstance(img, str) else self._encode_image(img)
                    ),
                    "detail": detail,
                }
                for img in images
            ]

            if isinstance(prompt, str):
                final_input = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            *image_contents,
                        ],
                    }
                ]
            else:
                final_input = [*prompt, {"role": "user", "content": image_contents}]
        else:
            final_input = prompt

        api_kwargs.update(
            {
                "model": model,
                "input": final_input,
                "text_format": response_model,
            }
        )

        try:
            response = await asyncio.to_thread(
                self.client.responses.parse, **api_kwargs
            )
            parsed = response.output_parsed
            # If output_parsed is not the expected type or is empty, fall back
            if not isinstance(parsed, response_model) or self._is_parsed_result_empty(
                parsed
            ):
                logger.info(
                    "[_process_one_structured_prompt] parsed empty/wrong type "
                    "(type={}, empty={}), falling back to _structured_via_chat, model={}",
                    type(parsed).__name__,
                    self._is_parsed_result_empty(parsed) if parsed else True,
                    model,
                )
                return await asyncio.to_thread(
                    self._structured_via_chat,
                    prompt,
                    response_model,
                    model,
                    image,
                    detail,
                    api_kwargs.get("instructions"),
                )
            return parsed
        except Exception as e:
            logger.warning(
                "[_process_one_structured_prompt] responses.parse failed ({}), "
                "falling back to _structured_via_chat, model={}",
                e,
                model,
            )
            return await asyncio.to_thread(
                self._structured_via_chat,
                prompt,
                response_model,
                model,
                image,
                detail,
                api_kwargs.get("instructions"),
            )

    async def structured_prompt_batch(
        self,
        prompts: List[Union[str, List[Dict[str, Any]]]],
        response_model: Type[BaseModel],
        model: str = "gpt-5.4",
        images: Optional[
            List[Optional[Union[str, np.ndarray, Image.Image, list]]]
        ] = None,
        detail: str = "auto",
        **kwargs: Any,
    ) -> List[Union[BaseModel, Exception]]:
        """Process multiple structured prompts concurrently."""
        if images and len(prompts) != len(images):
            raise ValueError("Number of prompts must match number of image entries.")

        tasks = [
            self._process_one_structured_prompt(
                prompt=prompt,
                response_model=response_model,
                model=model,
                image=images[i] if images else None,
                detail=detail,
                **kwargs,
            )
            for i, prompt in enumerate(prompts)
        ]

        if not tasks:
            return []
        return await asyncio.gather(*tasks, return_exceptions=True)

    def embed_text(
        self,
        input_text: Union[str, List[str]],
        model: str = "text-embedding-3-small",
        *,
        normalize: bool = False,
        dimensions: Optional[int] = None,
        encoding_format: str = "float",
        return_usage: bool = False,
        max_total_tokens: int = 300_000,
        split_aggregate: str = "mean",
    ) -> Union[List[float], List[List[float]], Dict[str, Any]]:
        """Create vector embeddings for text strings."""
        try:
            import tiktoken

            _enc_cache: Dict[str, Any] = {}

            def count_tokens(text: str) -> int:
                enc = _enc_cache.get(model)
                if enc is None:
                    try:
                        enc = tiktoken.encoding_for_model(model)
                    except Exception:
                        enc = tiktoken.get_encoding("cl100k_base")
                    _enc_cache[model] = enc
                return len(enc.encode(text))

        except Exception:

            def count_tokens(text: str) -> int:
                return max(1, len(text) // 4)

        single_input = isinstance(input_text, str)
        original_inputs = [input_text] if single_input else list(input_text)
        cleaned_inputs = [s.replace("\n", " ") for s in original_inputs]

        def split_large_text(text: str) -> List[str]:
            tokens = count_tokens(text)
            if tokens <= max_total_tokens:
                return [text]
            parts = [
                p.strip()
                for p in text.replace("?", ".").replace("!", ".").split(".")
                if p.strip()
            ]
            chunks: List[str] = []
            cur: List[str] = []
            cur_tokens = 0
            for sent in parts:
                sent_tokens = count_tokens(sent)
                if sent_tokens > max_total_tokens:
                    approx_chars = max_total_tokens * 4
                    for i in range(0, len(sent), approx_chars):
                        chunks.append(sent[i : i + approx_chars])
                    continue
                if cur_tokens + sent_tokens <= max_total_tokens:
                    cur.append(sent)
                    cur_tokens += sent_tokens
                else:
                    chunks.append(". ".join(cur))
                    cur = [sent]
                    cur_tokens = sent_tokens
            if cur:
                chunks.append(". ".join(cur))
            return chunks

        expanded_inputs: List[str] = []
        original_to_chunk_indices: List[List[int]] = []
        for text in cleaned_inputs:
            chunks = split_large_text(text)
            idx_list = [len(expanded_inputs) + i for i in range(len(chunks))]
            expanded_inputs.extend(chunks)
            original_to_chunk_indices.append(idx_list)

        # Batch by token count
        batches: List[List[int]] = []
        current_batch: List[int] = []
        current_tokens = 0
        for idx, txt in enumerate(expanded_inputs):
            tok = count_tokens(txt)
            if tok > max_total_tokens:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                batches.append([idx])
                continue
            if current_tokens + tok > max_total_tokens and current_batch:
                batches.append(current_batch)
                current_batch = [idx]
                current_tokens = tok
            else:
                current_batch.append(idx)
                current_tokens += tok
        if current_batch:
            batches.append(current_batch)

        all_chunk_embeddings: List[Optional[List[float]]] = [None] * len(
            expanded_inputs
        )
        usage_records: List[Any] = []
        model_name: Optional[str] = None

        for batch_indices in batches:
            batch_inputs = [expanded_inputs[i] for i in batch_indices]
            create_kwargs: Dict[str, Any] = {
                "model": model,
                "input": batch_inputs,
                "encoding_format": encoding_format,
            }
            if dimensions is not None:
                create_kwargs["dimensions"] = dimensions
            response = self.client.embeddings.create(**create_kwargs)
            model_name = response.model
            for local_i, emb in zip(
                batch_indices, [d.embedding for d in response.data]
            ):
                all_chunk_embeddings[local_i] = emb
            usage_records.append(getattr(response, "usage", None))

        # Aggregate chunks back to original inputs
        import numpy as _np

        def aggregate(emb_list: List[List[float]]) -> List[float]:
            if len(emb_list) == 1:
                return emb_list[0]
            arr = _np.asarray(emb_list, dtype=_np.float32)
            vec = arr.sum(axis=0) if split_aggregate == "sum" else arr.mean(axis=0)
            return vec.tolist()

        aggregated = [
            aggregate([all_chunk_embeddings[i] for i in idxs])
            for idxs in original_to_chunk_indices
        ]

        if normalize:

            def _l2_norm(vec: List[float]) -> List[float]:
                arr = _np.asarray(vec, dtype=_np.float32)
                norm = _np.linalg.norm(arr)
                return vec if norm == 0 else (arr / norm).tolist()

            aggregated = [_l2_norm(e) for e in aggregated]

        base_result = aggregated[0] if single_input else aggregated

        if return_usage:
            return {
                "embeddings": base_result,
                "usage": usage_records[0] if len(usage_records) == 1 else usage_records,
                "model": model_name,
            }
        return base_result
