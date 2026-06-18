from typing import List
import re
from .base import BaseObjectsExtractor


class SimpleSplitObjectsExtractor(BaseObjectsExtractor):
    def extract(self, prompt: str) -> List[str]:
        """
        "a yellow dog, a cute cat, a white lamp in front of the mirror, in a room..."
        return ["a yellow dog", "a cute cat", "a white lamp"]

        "a yellow dog, a cute cat, a white lamp, a red chair in front of the mirror, in a room..."
        return ["a yellow dog", "a cute cat", "a white lamp", "a red chair"]
        """
        # Extract only the text before "in front of [the/a] mirror"
        match = re.search(r'\s*in\s+front\s+of\s+(?:a|the)?\s*mirror', prompt, flags=re.IGNORECASE)
        before_mirror = prompt[:match.start()] if match else prompt

        # Split by comma to get individual objects
        items = [item.strip() for item in before_mirror.split(',')]
        return [item for item in items if item]


class SimpleSplitObjectsExtractor2(BaseObjectsExtractor):
    def extract(self, prompt: str) -> List[str]:
        # Remove the trailing clause:
        # "both the X and its/... reflection(s) are visible"
        
        # CHANGE: switched .+? to .*? before 'reflections' to allow 
        # for cases with NO adjectives (e.g., "its reflection")
        prompt = re.sub(
            r',\s*both\s+the\s+.+?\s+and\s+(?:its|their)\s+.*?reflections?\s+are\s+visible\s*$', 
            '', 
            prompt, 
            flags=re.IGNORECASE
        )

        # Split by comma
        items = [item.strip() for item in prompt.split(',')]

        # Remove "in front of a mirror/the mirror"
        cleaned_items = []
        for item in items:
            cleaned = re.sub(
                r'\s*in\s+front\s+of\s+(?:a|the)?\s*mirror\s*', 
                '', 
                item, 
                flags=re.IGNORECASE
            ).strip()

            if not cleaned:
                continue

            # Once we hit a scene/location description (e.g. "in a room...", "with warm sunlight..."),
            # stop — everything after it is part of the scene, not objects.
            if re.match(r'^(?:in|with|surrounded\s+by|during)\s+', cleaned, flags=re.IGNORECASE):
                break

            # Remove leading "a " or "an "
            cleaned = re.sub(r'^(?:a|an)\s+', '', cleaned, flags=re.IGNORECASE).strip()

            if cleaned:
                cleaned_items.append(cleaned)

        return cleaned_items
