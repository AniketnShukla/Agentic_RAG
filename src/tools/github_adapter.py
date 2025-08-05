import os
import re
from typing import List, Dict, Any, Optional

class GitHubAdapter:
    """
    An adapter for interacting with the GitHub REST API.
    Provides methods to fetch repository data like files, issues, and commits.
    """

    def __init__(self, token: Optional[str] = None):
        """
        Initializes the GitHubAdapter.

        Args:
            token: A GitHub personal access token for authentication.
                   If not provided, it will try to use the GITHUB_API_TOKEN environment variable.
        """
        self.token = token or os.getenv("GITHUB_API_TOKEN")
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
        }
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"

    def _parse_repo_url(self, repo_url: str) -> Optional[Dict[str, str]]:
        """
        Parses a GitHub repository URL to extract the owner and repo name.
        Example: "https://github.com/owner/repo" -> {"owner": "owner", "repo": "repo"}
        """
        match = re.match(r"https?://github\.com/([^/]+)/([^/]+)", repo_url)
        if match:
            return {"owner": match.group(1), "repo": match.group(2)}
        return None

    def fetch_repo_file(self, owner: str, repo: str, file_path: str = "README.md") -> Optional[str]:
        """
        Fetches the content of a specific file from a GitHub repository.

        NOTE: This is a placeholder implementation. It does not make a real API call.
        """
        print(f"--- MOCK GITHUB API CALL ---")
        print(f"Fetching file '{file_path}' from {owner}/{repo}")
        print(f"URL: https://api.github.com/repos/{owner}/{repo}/contents/{file_path}")
        print(f"Headers: {self.headers}")
        print(f"--------------------------")

        # In a real implementation, you would use a library like `requests` to make a GET request.
        # The response would be a JSON object containing the file content in base64.
        # You would then decode it and return the text content.
        # Example:
        # import requests
        # import base64
        # response = requests.get(f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}", headers=self.headers)
        # response.raise_for_status()
        # content = base64.b64decode(response.json()['content']).decode('utf-8')
        # return content

        return f"Mock content of {file_path} from {owner}/{repo}."

    def fetch_issues(self, owner: str, repo: str, state: str = "open") -> List[str]:
        """
        Fetches issues from a GitHub repository.

        NOTE: This is a placeholder implementation. It does not make a real API call.
        """
        print(f"--- MOCK GITHUB API CALL ---")
        print(f"Fetching '{state}' issues from {owner}/{repo}")
        print(f"URL: https://api.github.com/repos/{owner}/{repo}/issues?state={state}")
        print(f"--------------------------")

        # In a real implementation, you would handle pagination to get all issues.
        # The response is a list of issue objects. We would format them into strings.
        mock_issues = [
            {"title": "Bug: Fixes #123", "body": "This is a mock bug report."},
            {"title": "Feature: New dashboard", "body": "Add a new dashboard for analytics."},
        ]

        return [f"Issue: {issue['title']}\n{issue['body']}" for issue in mock_issues]

    def fetch_commits(self, owner: str, repo: str) -> List[str]:
        """
        Fetches recent commit messages from a GitHub repository.

        NOTE: This is a placeholder implementation. It does not make a real API call.
        """
        print(f"--- MOCK GITHUB API CALL ---")
        print(f"Fetching commits from {owner}/{repo}")
        print(f"URL: https://api.github.com/repos/{owner}/{repo}/commits")
        print(f"--------------------------")

        # In a real implementation, you would handle pagination.
        # The response is a list of commit objects. We extract the commit message.
        mock_commits = [
            {"commit": {"message": "feat: Add user authentication"}},
            {"commit": {"message": "fix: Correct typo in README"}},
        ]

        return [commit['commit']['message'] for commit in mock_commits]
