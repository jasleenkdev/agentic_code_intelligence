"""Demo example queries, hand-written in the style of APPS problems (not from the test split)."""

from __future__ import annotations

EXAMPLES: dict[str, list[dict[str, str]]] = {
    "apps": [
        {
            "label": "Max subarray sum",
            "query": "You are given an array of n integers a_1, ..., a_n. Find the maximum "
            "possible "
            "sum of a non-empty contiguous subarray.\n\n-----Input-----\nThe first line contains "
            "n (1 <= n <= 2*10^5). The second line contains n integers (|a_i| <= 10^9).\n\n"
            "-----Output-----\nPrint the maximum subarray sum.",
        },
        {
            "label": "Shortest path or -1",
            "query": "There are n cities and m bidirectional roads, each road connects two cities. "
            "Find the minimum number of roads needed to travel from city 1 to city n.\n\n"
            "-----Input-----\nThe first line contains n and m. Each of the next m lines contains "
            "two integers u and v.\n\n-----Output-----\nPrint the minimum number of roads, or -1 "
            "if city n cannot be reached.",
        },
        {
            "label": "Palindrome YES/NO",
            "query": "You are given t test cases. In each test case you are given a string s of "
            "lowercase Latin letters. Determine whether you can delete at most one character so "
            "that s becomes a palindrome. For each test case print YES or NO.",
        },
        {
            "label": "Pairs divisible by k",
            "query": "Given n integers and an integer k, count the number of pairs (i, j) with "
            "i < j such that a_i + a_j is divisible by k. Print the answer modulo 998244353.",
        },
        {
            "label": "Grid paths with obstacles",
            "query": "A robot stands in the top-left cell of an h x w grid. Some cells are walls "
            "('#'), others are empty ('.'). The robot moves only right or down. Count the number "
            "of paths to the bottom-right cell modulo 10^9+7.",
        },
        {
            "label": "Version hint (all versions)",
            "query": "Read two integers a and b and print their greatest common divisor "
            "(original version)",
        },
    ],
    "gladys": [
        {
            "label": "Purge old states",
            "query": "delete old device feature states from the database",
        },
        {"label": "Telegram message", "query": "send a message to a user through Telegram"},
        {
            "label": "Scene trigger",
            "query": "check whether a scene trigger condition matches an event",
        },
        {
            "label": "Version hint",
            "query": "register a proxy service for an external integration (latest version)",
        },
    ],
}
