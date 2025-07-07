# -*- coding: utf-8 -*-
# @Time    : 2025/01/07
# @File    : test_suggest_tags.py

import asyncio
from novelai_python.sdk.ai.generate_image import SuggestTags
from novelai_python.sdk.ai._enum import Model


def test_suggest_tags_default_model():
    async def run():
        suggest = SuggestTags(prompt="senko")
        result = await suggest.request()
        assert suggest.model == Model.NAI_DIFFUSION_4_5_FULL
        assert suggest.lang == "en"
        assert len(result.tags) > 0
        assert all(hasattr(tag, 'tag') for tag in result.tags)
    
    asyncio.run(run())


def test_suggest_tags_with_nai_diffusion_3():
    async def run():
        suggest = SuggestTags(
            prompt="senko",
            model=Model.NAI_DIFFUSION_3
        )
        result = await suggest.request()
        assert len(result.tags) > 0
    
    asyncio.run(run())