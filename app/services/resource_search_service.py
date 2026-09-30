"""
BodhaQ Resource Search Service.

Searches the web for high-quality learning resources and
YouTube learning videos related to a study topic.

Responsibilities:
    - Search the web for relevant educational resources.
    - Prefer authoritative and educational sources.
    - Search YouTube separately for learning videos.
    - Remove duplicate URLs.
    - Classify resources by type.
    - Rank resources by relevance and source quality.
    - Return small, frontend-friendly result lists.
    - Never ask Gemini to invent URLs.

Provider:
    - Tavily Search API

Security:
    - Tavily API keys are request-scoped.
    - API keys are never stored on the service singleton.
    - API keys are never persisted.
    - API keys are never logged.
    - API keys are never returned to the frontend.

The service is intentionally independent from Gemini so that web
retrieval can be replaced later without changing the Study/Learning
architecture.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

REQUEST_TIMEOUT_SECONDS = 10

DEFAULT_RESOURCE_COUNT = 8
MAX_RESOURCE_COUNT = 10

DEFAULT_VIDEO_COUNT = 8
MAX_VIDEO_COUNT = 10

MAX_SEARCH_RESULTS = 12

MAX_TOPIC_LENGTH = 300
MAX_TITLE_LENGTH = 300
MAX_DESCRIPTION_LENGTH = 500

MAX_PROCESSED_RESULTS = 50


# ============================================================================
# SOURCE QUALITY
# ============================================================================

DOMAIN_QUALITY: dict[str, int] = {
    # ------------------------------------------------------------------------
    # Official / primary documentation
    # ------------------------------------------------------------------------

    "docs.python.org": 100,
    "docs.oracle.com": 100,
    "developer.mozilla.org": 100,
    "learn.microsoft.com": 100,
    "docs.aws.amazon.com": 100,
    "kubernetes.io": 100,
    "docs.docker.com": 100,
    "fastapi.tiangolo.com": 100,
    "spring.io": 100,
    "react.dev": 100,
    "dev.java": 100,
    "cplusplus.com": 90,
    "go.dev": 100,
    "rust-lang.org": 100,
    "nodejs.org": 100,
    "postgresql.org": 100,
    "mysql.com": 95,
    "mongodb.com": 100,
    "terraform.io": 100,
    "developer.android.com": 100,

    # ------------------------------------------------------------------------
    # Strong educational platforms
    # ------------------------------------------------------------------------

    "geeksforgeeks.org": 85,
    "w3schools.com": 80,
    "freecodecamp.org": 90,
    "tutorialspoint.com": 75,
    "programiz.com": 85,
    "baeldung.com": 90,
    "realpython.com": 90,
    "digitalocean.com": 85,
    "educative.io": 85,
    "simplilearn.com": 75,
    "scaler.com": 75,
    "javatpoint.com": 75,

    # ------------------------------------------------------------------------
    # Practice platforms
    # ------------------------------------------------------------------------

    "leetcode.com": 90,
    "hackerrank.com": 90,
    "codeforces.com": 90,
    "codechef.com": 90,
    "codingninjas.com": 80,
    "atcoder.jp": 90,

    # ------------------------------------------------------------------------
    # Academic / reference
    # ------------------------------------------------------------------------

    "wikipedia.org": 70,
    "arxiv.org": 90,
    "acm.org": 90,
    "ieeexplore.ieee.org": 90,
    "stackoverflow.com": 75,

    # ------------------------------------------------------------------------
    # Video
    # ------------------------------------------------------------------------

    "youtube.com": 75,
    "youtu.be": 75,
}


LOW_QUALITY_DOMAINS = {
    "pinterest.com",
    "facebook.com",
    "instagram.com",
    "x.com",
    "twitter.com",
    "quora.com",
}


YOUTUBE_DOMAINS = {
    "youtube.com",
    "youtu.be",
}


# ============================================================================
# INTERNAL RESOURCE MODEL
# ============================================================================


@dataclass
class LearningResource:
    """
    Internal learning-resource representation.

    relevance_score is used only for backend ranking and is not exposed
    to the frontend.
    """

    title: str
    url: str
    description: str
    resource_type: str
    domain: str
    relevance_score: float


# ============================================================================
# PROVIDER EXCEPTIONS
# ============================================================================


class ResourceSearchAuthenticationError(RuntimeError):
    """Raised when Tavily rejects the supplied API key."""


class ResourceSearchQuotaError(RuntimeError):
    """Raised when Tavily rate limits the request."""


# ============================================================================
# SERVICE
# ============================================================================


class ResourceSearchService:
    """
    Search and rank online learning resources and videos.

    The service does not generate URLs itself.

    Every returned URL must originate from the configured search provider.

    Tavily API keys are supplied per request and are never stored as
    instance state.
    """

    # ========================================================================
    # PUBLIC API — RESOURCES
    # ========================================================================

    def search_resources(
        self,
        topic: str,
        api_key: str,
        max_results: int = DEFAULT_RESOURCE_COUNT,
    ) -> list[dict[str, Any]]:
        """
        Search for relevant non-video learning resources.

        YouTube results are intentionally excluded because videos have
        their own dedicated search endpoint and frontend section.

        Args:
            topic:
                Study topic to search for.

            api_key:
                Request-scoped Tavily API key.

            max_results:
                Maximum number of resources to return.
        """

        topic = self._clean_topic(topic)

        if not topic:
            raise ValueError(
                "Topic cannot be empty."
            )

        self._validate_topic_length(
            topic
        )

        api_key = self._validate_api_key(
            api_key
        )

        max_results = self._normalize_result_count(
            max_results,
            MAX_RESOURCE_COUNT,
        )

        try:
            raw_results = self._search_tavily(
                topic=topic,
                api_key=api_key,
                max_results=MAX_SEARCH_RESULTS,
                search_type="resources",
            )

            resources = self._process_results(
                topic=topic,
                results=raw_results,
                max_results=max_results,
                exclude_videos=True,
            )

            logger.info(
                "[Resources] Found %d resources.",
                len(resources),
            )

            return [
                self._resource_to_dict(resource)
                for resource in resources
            ]

        except (
            ResourceSearchAuthenticationError,
            ResourceSearchQuotaError,
        ):
            raise

        except ValueError:
            raise

        except RuntimeError:
            raise

        except Exception as exc:
            logger.exception(
                "[Resources] Unexpected resource-search failure."
            )

            raise RuntimeError(
                "Online resource search is temporarily unavailable."
            ) from exc

    # ========================================================================
    # PUBLIC API — VIDEOS
    # ========================================================================

    def search_videos(
        self,
        topic: str,
        api_key: str,
        max_results: int = DEFAULT_VIDEO_COUNT,
    ) -> list[dict[str, Any]]:
        """
        Search specifically for YouTube learning videos.

        The provider may not expose thumbnail, channel, or duration
        metadata. Those fields are therefore optional and are only
        populated when the provider actually returns them.

        Args:
            topic:
                Study topic to search for.

            api_key:
                Request-scoped Tavily API key.

            max_results:
                Maximum number of videos to return.
        """

        topic = self._clean_topic(topic)

        if not topic:
            raise ValueError(
                "Topic cannot be empty."
            )

        self._validate_topic_length(
            topic
        )

        api_key = self._validate_api_key(
            api_key
        )

        max_results = self._normalize_result_count(
            max_results,
            MAX_VIDEO_COUNT,
        )

        try:
            raw_results = self._search_tavily(
                topic=topic,
                api_key=api_key,
                max_results=MAX_SEARCH_RESULTS,
                search_type="videos",
            )

            videos = self._process_video_results(
                topic=topic,
                results=raw_results,
                max_results=max_results,
            )

            logger.info(
                "[Videos] Found %d videos.",
                len(videos),
            )

            return [
                self._video_to_dict(video)
                for video in videos
            ]

        except (
            ResourceSearchAuthenticationError,
            ResourceSearchQuotaError,
        ):
            raise

        except ValueError:
            raise

        except RuntimeError:
            raise

        except Exception as exc:
            logger.exception(
                "[Videos] Unexpected video-search failure."
            )

            raise RuntimeError(
                "Online video search is temporarily unavailable."
            ) from exc

    # ========================================================================
    # PUBLIC API — CONNECTION TEST
    # ========================================================================

    def test_connection(
        self,
        api_key: str,
    ) -> dict[str, Any]:
        """
        Validate a request-scoped Tavily API key.

        The connection test intentionally performs a minimal Tavily search
        rather than relying on a separate provider endpoint. This verifies
        that the supplied key is accepted by the same API used for normal
        resource searches.

        The API key remains request-scoped and is never stored or logged.
        """

        api_key = self._validate_api_key(api_key)

        payload = {
            "api_key": api_key,
            "query": "BodhaQ learning",
            "search_depth": "basic",
            "topic": "general",
            "max_results": 1,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }

        try:
            response = requests.post(
                TAVILY_SEARCH_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

        except requests.Timeout as exc:
            raise RuntimeError(
                "Tavily connection timed out."
            ) from exc

        except requests.RequestException as exc:
            raise RuntimeError(
                "Unable to connect to Tavily."
            ) from exc

        if response.status_code in {401, 403}:
            raise ResourceSearchAuthenticationError(
                "Tavily API key authentication failed."
            )

        if response.status_code == 429:
            raise ResourceSearchQuotaError(
                "Tavily API rate limit reached."
            )

        if response.status_code >= 500:
            raise RuntimeError(
                "Tavily service is temporarily unavailable."
            )

        if response.status_code >= 400:
            raise RuntimeError(
                "Tavily connection test failed."
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "Tavily returned an invalid response."
            ) from exc

        if not isinstance(data, dict):
            raise RuntimeError(
                "Tavily returned an invalid response."
            )

        return {
            "valid": True,
            "message": "Tavily connection successful.",
        }

    # ========================================================================
    # TAVILY
    # ========================================================================

    def _search_tavily(
        self,
        topic: str,
        api_key: str,
        max_results: int,
        search_type: str,
    ) -> list[dict[str, Any]]:
        """
        Search Tavily.

        search_type:
            resources
            videos
        """

        api_key = self._validate_api_key(
            api_key
        )

        if search_type == "videos":
            query = self._build_video_search_query(
                topic
            )

        elif search_type == "resources":
            query = self._build_search_query(
                topic
            )

        else:
            raise ValueError(
                "Invalid Tavily search type."
            )

        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "advanced",
            "topic": "general",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }

        try:
            response = requests.post(
                TAVILY_SEARCH_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

        except requests.Timeout as exc:
            raise RuntimeError(
                "Online search timed out."
            ) from exc

        except requests.RequestException as exc:
            raise RuntimeError(
                "Unable to connect to the online search service."
            ) from exc

        # --------------------------------------------------------------------
        # Provider status handling
        # --------------------------------------------------------------------

        if response.status_code in {401, 403}:
            raise ResourceSearchAuthenticationError(
                "Online search authentication failed."
            )

        if response.status_code == 429:
            raise ResourceSearchQuotaError(
                "Online search rate limit reached."
            )

        if response.status_code >= 500:
            raise RuntimeError(
                "Online search service is temporarily unavailable."
            )

        if response.status_code >= 400:
            logger.warning(
                "[Search] Tavily returned HTTP %s.",
                response.status_code,
            )

            raise RuntimeError(
                "Online search request failed."
            )

        # --------------------------------------------------------------------
        # Parse response
        # --------------------------------------------------------------------

        try:
            data = response.json()

        except ValueError as exc:
            raise RuntimeError(
                "Online search returned invalid data."
            ) from exc

        if not isinstance(
            data,
            dict,
        ):
            raise RuntimeError(
                "Online search returned an invalid response."
            )

        results = data.get(
            "results",
            [],
        )

        if not isinstance(
            results,
            list,
        ):
            logger.warning(
                "[Search] Tavily response contained an invalid "
                "'results' field."
            )

            return []

        # Provider responses are bounded before further processing.
        return [
            result
            for result in results[
                :MAX_PROCESSED_RESULTS
            ]
            if isinstance(
                result,
                dict,
            )
        ]

    # ========================================================================
    # QUERY BUILDING
    # ========================================================================

    @staticmethod
    def _build_search_query(
        topic: str,
    ) -> str:
        """
        Build a search query focused on educational material.
        """

        return (
            f"{topic} "
            "tutorial documentation guide examples "
            "learn reference"
        )

    @staticmethod
    def _build_video_search_query(
        topic: str,
    ) -> str:
        """
        Build a YouTube-focused learning query.
        """

        return (
            f"site:youtube.com {topic} "
            "tutorial lecture course explained "
            "programming learning"
        )

    # ========================================================================
    # RESOURCE PROCESSING
    # ========================================================================

    def _process_results(
        self,
        topic: str,
        results: list[dict[str, Any]],
        max_results: int,
        exclude_videos: bool = False,
    ) -> list[LearningResource]:
        """
        Clean, deduplicate, classify and rank search results.
        """

        candidates: list[LearningResource] = []
        seen_urls: set[str] = set()

        for result in results:

            if not isinstance(
                result,
                dict,
            ):
                continue

            url = self._clean_url(
                result.get("url")
            )

            if not url:
                continue

            normalized_url = self._normalize_url(
                url
            )

            if normalized_url in seen_urls:
                continue

            seen_urls.add(
                normalized_url
            )

            title = self._clean_text(
                result.get("title")
            )

            content = self._clean_text(
                result.get("content")
            )

            if not title:
                continue

            domain = self._extract_domain(
                url
            )

            if not domain:
                continue

            if self._is_blocked_domain(
                domain
            ):
                continue

            if (
                exclude_videos
                and domain in YOUTUBE_DOMAINS
            ):
                continue

            resource_type = self._classify_resource(
                url=url,
                title=title,
                content=content,
                domain=domain,
            )

            score = self._calculate_score(
                topic=topic,
                title=title,
                content=content,
                domain=domain,
                resource_type=resource_type,
                provider_score=result.get(
                    "score"
                ),
            )

            description = self._build_description(
                content
            )

            candidates.append(
                LearningResource(
                    title=title[
                        :MAX_TITLE_LENGTH
                    ],
                    url=url,
                    description=description[
                        :MAX_DESCRIPTION_LENGTH
                    ],
                    resource_type=resource_type,
                    domain=domain,
                    relevance_score=score,
                )
            )

        candidates.sort(
            key=lambda resource: (
                resource.relevance_score
            ),
            reverse=True,
        )

        candidates = self._diversify_resources(
            candidates,
            max_results,
        )

        return candidates[
            :max_results
        ]

    # ========================================================================
    # VIDEO PROCESSING
    # ========================================================================

    def _process_video_results(
        self,
        topic: str,
        results: list[dict[str, Any]],
        max_results: int,
    ) -> list[dict[str, Any]]:
        """
        Clean, validate, deduplicate and rank YouTube results.
        """

        candidates: list[dict[str, Any]] = []
        seen_urls: set[str] = set()

        for result in results:

            if not isinstance(
                result,
                dict,
            ):
                continue

            url = self._clean_url(
                result.get("url")
            )

            if not url:
                continue

            domain = self._extract_domain(
                url
            )

            if domain not in YOUTUBE_DOMAINS:
                continue

            normalized_url = self._normalize_url(
                url
            )

            if normalized_url in seen_urls:
                continue

            seen_urls.add(
                normalized_url
            )

            title = self._clean_text(
                result.get("title")
            )

            if not title:
                continue

            description = self._build_description(
                self._clean_text(
                    result.get("content")
                )
            )

            score = self._calculate_video_score(
                topic=topic,
                title=title,
                description=description,
                provider_score=result.get(
                    "score"
                ),
            )

            # Tavily may expose metadata in provider-specific fields.
            # Only use metadata when it is actually present.
            thumbnail = self._optional_string(
                result.get("thumbnail")
            )

            channel = self._optional_string(
                result.get("channel")
            )

            duration = self._optional_string(
                result.get("duration")
            )

            candidates.append(
                {
                    "title": title[
                        :MAX_TITLE_LENGTH
                    ],
                    "url": url,
                    "thumbnail": thumbnail,
                    "channel": channel,
                    "description": description[
                        :MAX_DESCRIPTION_LENGTH
                    ],
                    "duration": duration,
                    "_score": score,
                }
            )

        candidates.sort(
            key=lambda video: video.get(
                "_score",
                0.0,
            ),
            reverse=True,
        )

        return candidates[
            :max_results
        ]

    # ========================================================================
    # SCORING
    # ========================================================================

    def _calculate_score(
        self,
        topic: str,
        title: str,
        content: str,
        domain: str,
        resource_type: str,
        provider_score: Any,
    ) -> float:
        """
        Calculate an internal ranking score.

        This score is NOT shown to the user.
        """

        score = 0.0

        topic_tokens = self._tokenize(
            topic
        )

        title_tokens = self._tokenize(
            title
        )

        content_tokens = self._tokenize(
            content
        )

        if topic_tokens:

            title_matches = sum(
                1
                for token in topic_tokens
                if token in title_tokens
            )

            content_matches = sum(
                1
                for token in topic_tokens
                if token in content_tokens
            )

            title_relevance = (
                title_matches
                / len(topic_tokens)
            )

            content_relevance = min(
                content_matches
                / len(topic_tokens),
                1.0,
            )

            score += (
                title_relevance * 45
            )

            score += (
                content_relevance * 25
            )

        if isinstance(
            provider_score,
            (int, float),
        ) and not isinstance(
            provider_score,
            bool,
        ):
            score += (
                max(
                    0.0,
                    min(
                        float(provider_score),
                        1.0,
                    ),
                )
                * 15
            )

        domain_score = self._domain_quality(
            domain
        )

        score += (
            domain_score * 0.12
        )

        type_bonus = {
            "official_documentation": 10,
            "tutorial": 7,
            "practice": 7,
            "video": 5,
            "reference": 5,
            "article": 3,
            "community": 1,
            "other": 0,
        }

        score += type_bonus.get(
            resource_type,
            0,
        )

        return round(
            score,
            2,
        )

    def _calculate_video_score(
        self,
        topic: str,
        title: str,
        description: str,
        provider_score: Any,
    ) -> float:
        """
        Calculate an internal ranking score for videos.
        """

        score = 0.0

        topic_tokens = self._tokenize(
            topic
        )

        title_tokens = self._tokenize(
            title
        )

        description_tokens = self._tokenize(
            description
        )

        if topic_tokens:

            title_matches = sum(
                1
                for token in topic_tokens
                if token in title_tokens
            )

            description_matches = sum(
                1
                for token in topic_tokens
                if token in description_tokens
            )

            score += (
                title_matches
                / len(topic_tokens)
            ) * 60

            score += min(
                description_matches
                / len(topic_tokens),
                1.0,
            ) * 20

        if isinstance(
            provider_score,
            (int, float),
        ) and not isinstance(
            provider_score,
            bool,
        ):
            score += (
                max(
                    0.0,
                    min(
                        float(provider_score),
                        1.0,
                    ),
                )
                * 20
            )

        return round(
            score,
            2,
        )

    # ========================================================================
    # RESOURCE CLASSIFICATION
    # ========================================================================

    def _classify_resource(
        self,
        url: str,
        title: str,
        content: str,
        domain: str,
    ) -> str:
        """
        Determine the resource type.
        """

        combined = (
            f"{title} {content} {url}"
        ).lower()

        if domain in YOUTUBE_DOMAINS:
            return "video"

        practice_domains = {
            "leetcode.com",
            "hackerrank.com",
            "codeforces.com",
            "codechef.com",
            "atcoder.jp",
            "codingninjas.com",
        }

        if domain in practice_domains:
            return "practice"

        practice_keywords = (
            "practice problems",
            "coding problems",
            "exercises",
            "challenges",
            "problems",
        )

        if any(
            keyword in combined
            for keyword in practice_keywords
        ):
            return "practice"

        official_domains = {
            "docs.python.org",
            "docs.oracle.com",
            "developer.mozilla.org",
            "learn.microsoft.com",
            "docs.aws.amazon.com",
            "kubernetes.io",
            "docs.docker.com",
            "fastapi.tiangolo.com",
            "spring.io",
            "react.dev",
            "dev.java",
            "go.dev",
            "rust-lang.org",
            "nodejs.org",
            "postgresql.org",
            "mysql.com",
            "mongodb.com",
            "terraform.io",
            "developer.android.com",
        }

        if domain in official_domains:
            return "official_documentation"

        documentation_keywords = (
            "documentation",
            "docs",
            "api reference",
            "language specification",
            "official",
        )

        if (
            self._domain_quality(domain) >= 95
            and any(
                keyword in combined
                for keyword in documentation_keywords
            )
        ):
            return "official_documentation"

        tutorial_keywords = (
            "tutorial",
            "learn",
            "getting started",
            "beginner",
            "guide",
            "how to",
        )

        if any(
            keyword in combined
            for keyword in tutorial_keywords
        ):
            return "tutorial"

        reference_keywords = (
            "reference",
            "api",
            "syntax",
            "documentation",
        )

        if any(
            keyword in combined
            for keyword in reference_keywords
        ):
            return "reference"

        if domain == "stackoverflow.com":
            return "community"

        article_keywords = (
            "article",
            "explained",
            "concept",
        )

        if any(
            keyword in combined
            for keyword in article_keywords
        ):
            return "article"

        return "other"

    # ========================================================================
    # DIVERSIFICATION
    # ========================================================================

    @staticmethod
    def _diversify_resources(
        resources: list[LearningResource],
        max_results: int,
    ) -> list[LearningResource]:
        """
        Prevent the Study page from being filled with the same type
        or domain.
        """

        if not resources:
            return []

        selected: list[
            LearningResource
        ] = []

        domain_counts: dict[
            str,
            int,
        ] = {}

        type_counts: dict[
            str,
            int,
        ] = {}

        # --------------------------------------------------------------------
        # First pass: diversity
        # --------------------------------------------------------------------

        for resource in resources:

            if len(selected) >= max_results:
                break

            domain_count = domain_counts.get(
                resource.domain,
                0,
            )

            type_count = type_counts.get(
                resource.resource_type,
                0,
            )

            if domain_count >= 3:
                continue

            if type_count >= 3:
                continue

            selected.append(
                resource
            )

            domain_counts[
                resource.domain
            ] = domain_count + 1

            type_counts[
                resource.resource_type
            ] = type_count + 1

        # --------------------------------------------------------------------
        # Second pass: fill remaining slots
        # --------------------------------------------------------------------

        if len(selected) < max_results:

            selected_urls = {
                resource.url
                for resource in selected
            }

            for resource in resources:

                if len(selected) >= max_results:
                    break

                if resource.url in selected_urls:
                    continue

                selected.append(
                    resource
                )

                selected_urls.add(
                    resource.url
                )

        return selected

    # ========================================================================
    # URL HELPERS
    # ========================================================================

    @staticmethod
    def _clean_url(
        value: Any,
    ) -> str:
        """
        Accept only HTTP(S) URLs.
        """

        if not isinstance(
            value,
            str,
        ):
            return ""

        url = value.strip()

        if not url:
            return ""

        parsed = urlparse(
            url
        )

        if parsed.scheme not in {
            "http",
            "https",
        }:
            return ""

        if not parsed.netloc:
            return ""

        return url

    @staticmethod
    def _normalize_url(
        url: str,
    ) -> str:
        """
        Normalize URL for duplicate detection.
        """

        parsed = urlparse(
            url
        )

        domain = (
            parsed.netloc
            .lower()
            .split(":")[0]
            .removeprefix("www.")
        )

        path = (
            parsed.path.rstrip("/")
            or "/"
        )

        query = parsed.query

        if query:
            return (
                f"{domain}"
                f"{path}"
                f"?{query}"
            )

        return (
            f"{domain}"
            f"{path}"
        )

    @staticmethod
    def _extract_domain(
        url: str,
    ) -> str:
        """
        Extract normalized hostname.
        """

        try:
            parsed = urlparse(
                url
            )

            return (
                parsed.netloc
                .lower()
                .split(":")[0]
                .removeprefix("www.")
            )

        except Exception:
            return ""

    # ========================================================================
    # DOMAIN HELPERS
    # ========================================================================

    @staticmethod
    def _domain_quality(
        domain: str,
    ) -> int:
        """
        Return source quality score.
        """

        domain = domain.lower()

        if domain in DOMAIN_QUALITY:
            return DOMAIN_QUALITY[
                domain
            ]

        for known_domain, score in DOMAIN_QUALITY.items():

            if domain.endswith(
                f".{known_domain}"
            ):
                return score

        return 50

    @staticmethod
    def _is_blocked_domain(
        domain: str,
    ) -> bool:
        """
        Exclude social/spam-oriented domains.
        """

        domain = domain.lower()

        if domain in LOW_QUALITY_DOMAINS:
            return True

        for blocked in LOW_QUALITY_DOMAINS:

            if domain.endswith(
                f".{blocked}"
            ):
                return True

        return False

    # ========================================================================
    # TEXT HELPERS
    # ========================================================================

    @staticmethod
    def _clean_text(
        value: Any,
    ) -> str:
        """
        Clean arbitrary provider text.
        """

        if not isinstance(
            value,
            str,
        ):
            return ""

        return re.sub(
            r"\s+",
            " ",
            value,
        ).strip()

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> str | None:
        """
        Return a cleaned optional string.

        Unknown/missing provider metadata remains None.
        """

        if not isinstance(
            value,
            str,
        ):
            return None

        value = value.strip()

        return value or None

    @staticmethod
    def _build_description(
        content: str,
        max_length: int = 220,
    ) -> str:
        """
        Create a short resource description.
        """

        content = ResourceSearchService._clean_text(
            content
        )

        if not content:
            return (
                "Relevant learning resource for this topic."
            )

        if len(content) <= max_length:
            return content

        shortened = content[
            :max_length
        ]

        last_space = shortened.rfind(
            " "
        )

        if last_space > 100:
            shortened = shortened[
                :last_space
            ]

        return shortened + "..."

    @staticmethod
    def _clean_topic(
        topic: Any,
    ) -> str:
        """
        Normalize topic input.
        """

        if not isinstance(
            topic,
            str,
        ):
            return ""

        return re.sub(
            r"\s+",
            " ",
            topic,
        ).strip()

    @staticmethod
    def _tokenize(
        text: str,
    ) -> set[str]:
        """
        Basic tokenization for lightweight relevance scoring.
        """

        if not text:
            return set()

        tokens = re.findall(
            r"[a-zA-Z0-9+#.-]+",
            text.lower(),
        )

        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "for",
            "to",
            "of",
            "in",
            "on",
            "with",
            "is",
            "are",
            "from",
            "how",
            "what",
            "why",
            "using",
            "use",
            "learn",
            "guide",
            "tutorial",
        }

        return {
            token
            for token in tokens
            if token not in stop_words
            and len(token) >= 2
        }

    # ========================================================================
    # API KEY VALIDATION
    # ========================================================================

    @staticmethod
    def _validate_api_key(
        api_key: Any,
    ) -> str:
        """
        Validate the request-scoped Tavily API key.

        The key is intentionally returned only to the current call chain.
        It is never stored on the service instance.
        """

        if not isinstance(
            api_key,
            str,
        ):
            raise ValueError(
                "Tavily API key cannot be empty."
            )

        normalized = api_key.strip()

        if not normalized:
            raise ValueError(
                "Tavily API key cannot be empty."
            )

        return normalized

    # ========================================================================
    # INPUT LIMITS
    # ========================================================================

    @staticmethod
    def _validate_topic_length(
        topic: str,
    ) -> None:
        """
        Prevent excessively large external search queries.
        """

        if len(topic) > MAX_TOPIC_LENGTH:
            raise ValueError(
                "Topic exceeds the maximum allowed length of "
                f"{MAX_TOPIC_LENGTH} characters."
            )

    # ========================================================================
    # RESULT COUNT
    # ========================================================================

    @staticmethod
    def _normalize_result_count(
        count: int,
        maximum: int,
    ) -> int:
        """
        Validate and normalize result count.
        """

        if isinstance(
            count,
            bool,
        ):
            raise ValueError(
                "max_results must be an integer."
            )

        if not isinstance(
            count,
            int,
        ):
            raise ValueError(
                "max_results must be an integer."
            )

        if count < 1 or count > maximum:
            raise ValueError(
                f"max_results must be between 1 and "
                f"{maximum}."
            )

        return count

    # ========================================================================
    # RESULT VALIDATION
    # ========================================================================

    @staticmethod
    def _is_valid_resource(
        resource: LearningResource,
    ) -> bool:
        """
        Validate an internal resource before exposing it.
        """

        return (
            isinstance(
                resource.title,
                str,
            )
            and bool(
                resource.title.strip()
            )
            and isinstance(
                resource.url,
                str,
            )
            and bool(
                resource.url.strip()
            )
            and isinstance(
                resource.description,
                str,
            )
            and isinstance(
                resource.resource_type,
                str,
            )
            and isinstance(
                resource.domain,
                str,
            )
            and bool(
                resource.domain.strip()
            )
        )

    # ========================================================================
    # SERIALIZATION
    # ========================================================================

    @staticmethod
    def _resource_to_dict(
        resource: LearningResource,
    ) -> dict[str, Any]:
        """
        Convert internal resource model to API response format.

        relevance_score is intentionally omitted.
        """

        if not ResourceSearchService._is_valid_resource(
            resource
        ):
            raise ValueError(
                "Invalid learning resource."
            )

        return {
            "title": resource.title,
            "url": resource.url,
            "description": resource.description,
            "resource_type": resource.resource_type,
            "domain": resource.domain,
        }

    @staticmethod
    def _video_to_dict(
        video: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Convert internal video representation to API response format.

        Internal ranking information is intentionally omitted.
        """

        title = ResourceSearchService._clean_text(
            video.get("title")
        )

        url = ResourceSearchService._clean_url(
            video.get("url")
        )

        if not title or not url:
            raise ValueError(
                "Invalid learning video."
            )

        return {
            "title": title[
                :MAX_TITLE_LENGTH
            ],
            "url": url,
            "thumbnail": (
                ResourceSearchService._optional_string(
                    video.get("thumbnail")
                )
            ),
            "channel": (
                ResourceSearchService._optional_string(
                    video.get("channel")
                )
            ),
            "description": (
                ResourceSearchService._clean_text(
                    video.get("description")
                )[
                    :MAX_DESCRIPTION_LENGTH
                ]
            ),
            "duration": (
                ResourceSearchService._optional_string(
                    video.get("duration")
                )
            ),
        }


# ============================================================================
# SINGLETON
# ============================================================================

resource_search_service = ResourceSearchService()