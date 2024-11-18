import urllib3
import json
import os
from typing import Optional, Dict, Any, List

class SimpleAnthropicClient:
    """A simple client for the Anthropic API using urllib3."""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20240620"):
        """
        Initialize the client with API key and model.
        
        Args:
            api_key: Your Anthropic API key
            model: The model to use (defaults to claude-3-opus-20240229)
        """
        self.api_key = api_key
        self.model = model
        self.http = urllib3.PoolManager()
        self.base_url = "https://api.anthropic.com/v1"
        
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make a request to the Anthropic API."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        url = f"{self.base_url}/{endpoint}"
        
        response = self.http.request(
            method,
            url,
            headers=headers,
            body=json.dumps(data) if data else None
        )
        
        if response.status != 200:
            raise Exception(f"API request failed with status {response.status}: {response.data.decode()}")
            
        return json.loads(response.data.decode())
    
    def create_message(
        self,
        content: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a message using the Anthropic API.
        
        Args:
            content: The message content
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            system: Optional system message
        
        Returns:
            API response as a dictionary
        """
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if system:
            data["system"] = system
            
        return self._make_request("POST", "messages", data=data)



def main():
    result = lambda_handler(None, None)
    print("Lambda handler result:")
    print(result)

def lambda_handler(event, context):    
    # Get API key from environment variable
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    # Initialize client
    client = SimpleAnthropicClient(api_key)
    
    # Send a message
    try:
        response = client.create_message(
            content="Tell me a joke about programming.",
            max_tokens=100,
            temperature=0.7
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
if __name__ == "__main__":
    main()