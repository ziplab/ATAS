import asyncio
import json
from dataclasses import dataclass, field

import pytest
from atas.review import editorial
from atas.review.repair import repair_until_reviewed, RepairExhausted, promote_reviewed
from atas.review.film_review import file_hash
@dataclass
class Shot:
    prompt: str = 'Alice raises her hand in the room'
    raw: dict = field(default_factory=dict)






@pytest.mark.parametrize('same_scene',[0,1,'true'])
def test_comparison_requires_boolean_scene_evidence(same_scene):
    with pytest.raises(ValueError):
        editorial.validate_comparison(dict(ranking=[0],reason='best',same_scene=same_scene,
            actual_event='observed'),1)
