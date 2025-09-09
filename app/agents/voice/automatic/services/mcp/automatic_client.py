import httpx
import json
import base64
from typing import Dict, Any, Optional, Callable

from app.core.config import MCP_CLIENT_TIMEOUT, MAX_MCP_TOOLS
from app.core.logger import logger
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.schemas.function_schema import FunctionSchema
from app.agents.voice.automatic.types.models import (
    JSONRPCResponse,
    ToolCallResult,
    MCPTool
)

class StreamableHTTPTransport:
    """Handles JSON-RPC 2.0 over streaming HTTP with custom headers."""
    def __init__(self, server_url: str, auth_token: str, context: Dict[str, Any]):
        logger.debug(f"StreamableHTTPTransport initialized with server_url: '{server_url}'")
        if not server_url or not isinstance(server_url, str):
            raise ValueError("MCP server URL must be a non-empty string.")

        self._server_url = server_url.strip()
        self._auth_token = auth_token
        self._context = context  # Store original context for debugging
        self._context_b64 = base64.b64encode(json.dumps(context).encode()).decode()
        self._client = httpx.AsyncClient(timeout=MCP_CLIENT_TIMEOUT)
        self._demo_mode = context.get("enableDemoMode", False)
        
        logger.info(f"MCP Client Context being sent to neurolink: {json.dumps(context, indent=2)}")
        logger.debug(f"Base64 encoded x-context header: {self._context_b64}")
        
        # Decode and show what the server will receive
        try:
            decoded_context = json.loads(base64.b64decode(self._context_b64).decode())
            logger.info(f"Decoded context (what server receives): {json.dumps(decoded_context, indent=2)}")
        except Exception as e:
            logger.error(f"Failed to decode context for verification: {e}")
        
        logger.debug(f"Context being sent to neurolink: {context}")
        logger.debug(f"Base64 encoded context: {self._context_b64}")

    async def post(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Performs a JSON-RPC POST request and handles streaming response."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "x-context": self._context_b64,
        }
        if self._auth_token:
            headers["x-auth-token"] = self._auth_token
        json_rpc_payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
        
        query_params = {}
        if self._demo_mode:
            query_params["demoMode"] = "true"

        try:
            logger.info(f"Attempting to POST to: {self._server_url} with method: {method}")
            logger.debug(f"JSON-RPC payload: {json_rpc_payload}")
            logger.debug(f"Headers being sent: {headers}")
            async with self._client.stream("POST", self._server_url, headers=headers, json=json_rpc_payload, params=query_params) as response:
                logger.info(f"Response status: {response.status_code}")
                logger.info(f"Response headers: {dict(response.headers)}")
                logger.info(f"Response content type: {response.headers.get('content-type', 'Not specified')}")
                
                if response.is_error:
                    await response.aread()
                    response.raise_for_status()

                async for line in response.aiter_lines():
                    logger.debug(f"Received line from server: {line}")
                    if line.startswith("data:"):
                        json_str = line[len("data:"):].strip()
                        logger.info(f"Extracted JSON string from stream: {json_str}")
                        try:
                            # First, try to parse as raw JSON to see what we're getting
                            raw_data = json.loads(json_str)
                            logger.info(f"Raw response data: {raw_data}")
                            logger.info(f"Raw response keys: {list(raw_data.keys()) if isinstance(raw_data, dict) else 'Not a dict'}")
                            
                            # Check if this is a neurolink-specific event format
                            if isinstance(raw_data, dict) and "event" in raw_data:
                                event_type = raw_data.get("event")
                                logger.info(f"Received neurolink event: {event_type}")
                                
                                # Skip status/progress events and wait for actual data
                                if event_type in ["request_received", "processing", "progress"]:
                                    logger.debug(f"Skipping status event: {event_type}")
                                    continue
                                
                                # Check if this contains tools data anywhere in the response
                                def find_tools_in_response(data):
                                    """Recursively search for tools array in the response data"""
                                    if isinstance(data, dict):
                                        # Check direct tools key
                                        if "tools" in data and isinstance(data["tools"], list):
                                            return data["tools"]
                                        # Check in result key
                                        if "result" in data and isinstance(data["result"], dict):
                                            if "tools" in data["result"] and isinstance(data["result"]["tools"], list):
                                                return data["result"]["tools"]
                                        # Recursively search in all values
                                        for key, value in data.items():
                                            tools = find_tools_in_response(value)
                                            if tools:
                                                return tools
                                    elif isinstance(data, list):
                                        # Check if this is already a tools array
                                        if all(isinstance(item, dict) and "name" in item for item in data):
                                            return data
                                        # Recursively search in list items
                                        for item in data:
                                            tools = find_tools_in_response(item)
                                            if tools:
                                                return tools
                                    return None
                                
                                tools_array = find_tools_in_response(raw_data)
                                if tools_array:
                                    logger.info(f"Found {len(tools_array)} tools in neurolink response")
                                    logger.debug(f"First few tools: {tools_array[:3] if len(tools_array) > 3 else tools_array}")
                                    # Convert neurolink format to JSON-RPC format
                                    converted_response = {
                                        "jsonrpc": "2.0",
                                        "id": 1,
                                        "result": {"tools": tools_array}
                                    }
                                    logger.info(f"Converted to JSON-RPC format with {len(tools_array)} tools")
                                    return converted_response
                                
                                # Check if this is a call response
                                if event_type == "tool_response" or "result" in raw_data:
                                    logger.info("Found tool call result in neurolink response")
                                    # Convert neurolink format to JSON-RPC format
                                    converted_response = {
                                        "jsonrpc": "2.0", 
                                        "id": 1,
                                        "result": {
                                            "content": [{"type": "text", "text": raw_data.get("result", raw_data)}]
                                        }
                                    }
                                    logger.info(f"Converted tool result to JSON-RPC format: {converted_response}")
                                    return converted_response
                                
                                # If it's an error event
                                if event_type == "error" or "error" in raw_data:
                                    logger.error(f"Received error from neurolink: {raw_data}")
                                    converted_response = {
                                        "jsonrpc": "2.0",
                                        "id": 1,
                                        "error": {
                                            "code": -1,
                                            "message": str(raw_data.get("error", raw_data.get("message", "Unknown error")))
                                        }
                                    }
                                    return converted_response
                            
                            # If it's already in JSON-RPC format, validate it
                            elif "jsonrpc" in raw_data:
                                validated_response = JSONRPCResponse.model_validate_json(json_str)
                                response_dict = validated_response.model_dump(by_alias=True, exclude_none=True)

                                if isinstance(validated_response.result, ToolCallResult):
                                    for i, item in enumerate(validated_response.result.content):
                                        response_dict["result"]["content"][i]["text"] = item.text

                                return response_dict
                            
                            # If it's not an event format but might contain tools data, search for tools
                            else:
                                def find_tools_in_response(data):
                                    """Recursively search for tools array in the response data"""
                                    if isinstance(data, dict):
                                        # Check direct tools key
                                        if "tools" in data and isinstance(data["tools"], list):
                                            return data["tools"]
                                        # Check in result key
                                        if "result" in data and isinstance(data["result"], dict):
                                            if "tools" in data["result"] and isinstance(data["result"]["tools"], list):
                                                return data["result"]["tools"]
                                        # Recursively search in all values
                                        for key, value in data.items():
                                            tools = find_tools_in_response(value)
                                            if tools:
                                                return tools
                                    elif isinstance(data, list):
                                        # Check if this is already a tools array
                                        if all(isinstance(item, dict) and "name" in item for item in data):
                                            return data
                                        # Recursively search in list items
                                        for item in data:
                                            tools = find_tools_in_response(item)
                                            if tools:
                                                return tools
                                    return None
                                
                                tools_array = find_tools_in_response(raw_data)
                                if tools_array:
                                    logger.info(f"Found {len(tools_array)} tools in non-event response")
                                    logger.debug(f"First few tools: {tools_array[:3] if len(tools_array) > 3 else tools_array}")
                                    # Convert to JSON-RPC format
                                    converted_response = {
                                        "jsonrpc": "2.0",
                                        "id": 1,
                                        "result": {"tools": tools_array}
                                    }
                                    logger.info(f"Converted non-event response to JSON-RPC format with {len(tools_array)} tools")
                                    return converted_response
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to decode JSON from stream: {json_str}")
                            logger.error(f"JSON decode error: {e}")
                            raise ValueError("Received malformed JSON from server.")
                        except Exception as e: # Catches Pydantic's ValidationError
                            logger.error(f"Response validation failed: {e}")
                            logger.error(f"Trying to validate JSON: {json_str}")
                            # Don't raise immediately, continue processing other events
                            logger.warning(f"Continuing to process other events after validation error: {e}")
                            continue
                    else:
                        logger.debug(f"Skipping non-data line: {line}")

                logger.warning("Server stream ended without sending expected data. This might be a neurolink server that requires different handling.")
                # Return empty tools response for now to allow graceful fallback
                return {
                    "jsonrpc": "2.0",
                    "id": 1, 
                    "result": {"tools": []}
                }

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error on method {method}: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"HTTP Error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            logger.error(f"Network request error on method {method}: {e}")
            raise RuntimeError(f"Network Error: {e}") from e
        except Exception as e:
            logger.error(f"An unexpected transport error occurred on method {method}: {e}")
            raise

    async def close(self):
        await self._client.aclose()

class MCPClient:
    """A service to list, register, and call tools from a remote MCP server."""
    def __init__(self, server_url: str, auth_token: str, context: Dict[str, Any]):
        self._transport = StreamableHTTPTransport(server_url, auth_token, context)
        self._llm = None

    async def register_tools(self, llm, selective_functions) -> ToolsSchema:
        """Lists tools and registers them with the given LLM processor."""
        self._llm = llm
        logger.info("Registering tools from custom MCP client...")
        selective_functions_set = set(selective_functions)
        try:
            response_dict = await self._transport.post(method="tools/list")
            
            if response_dict.get("error"):
                error_details = response_dict['error']
                logger.error(f"Received JSON-RPC error when listing tools: {error_details}")
                raise RuntimeError(f"JSON-RPC Error listing tools: {error_details}")

            if not response_dict.get("result") or not response_dict["result"].get("tools"):
                logger.warning("Tool registration response was successful but contained no tools.")
                return ToolsSchema(standard_tools=[])

            raw_tools = response_dict["result"]["tools"]
            logger.info(f"Received {len(raw_tools)} tools from MCP server")
            
            # OpenAI has a limit of 128 tools, so we need to filter/prioritize
            MAX_TOOLS = MAX_MCP_TOOLS
            
            selective_tools_to_register = []
            if len(selective_functions) > 0:
                # If selective functions are specified, use only those
                for tool_data in raw_tools:
                    tool_name = tool_data["name"]
                    if tool_name in selective_functions_set:
                        selective_tools_to_register.append(tool_data)
                logger.info(f"Found {len(selective_tools_to_register)} tools matching selective functions")
                        
            tools_to_process = raw_tools
            if len(selective_tools_to_register) > 0:
                tools_to_process = selective_tools_to_register
            
            # If we still have too many tools, apply prioritization
            if len(tools_to_process) > MAX_TOOLS:
                logger.warning(f"Too many tools ({len(tools_to_process)}) for OpenAI limit ({MAX_TOOLS}). Applying prioritization.")
                
                # Prioritize tools based on categories - GA4 tools get highest priority
                ga4_tools = []
                high_priority_tools = []
                medium_priority_tools = []
                low_priority_tools = []
                
                high_priority_keywords = ["breeze", "analytics", "payment", "order", "shop"]
                medium_priority_keywords = ["browser", "playwright", "screenshot", "click", "type"]
                
                for tool in tools_to_process:
                    tool_name_lower = tool["name"].lower()
                    description_lower = (tool.get("description") or "").lower()
                    
                    # GA4 tools get highest priority
                    if "ga4" in tool_name_lower:
                        ga4_tools.append(tool)
                        logger.info(f"GA4 tool added to highest priority: {tool['name']}")
                    # Check if tool matches high priority keywords
                    elif any(keyword in tool_name_lower or keyword in description_lower for keyword in high_priority_keywords):
                        high_priority_tools.append(tool)
                    # Check if tool matches medium priority keywords
                    elif any(keyword in tool_name_lower or keyword in description_lower for keyword in medium_priority_keywords):
                        medium_priority_tools.append(tool)
                    else:
                        low_priority_tools.append(tool)
                
                # Combine tools in priority order: GA4 first, then high, medium, low
                prioritized_tools = ga4_tools + high_priority_tools + medium_priority_tools + low_priority_tools
                
                logger.info(f"Prioritization results: GA4={len(ga4_tools)}, High={len(high_priority_tools)}, Medium={len(medium_priority_tools)}, Low={len(low_priority_tools)}")
                
                tools_to_process = prioritized_tools[:MAX_TOOLS]
                logger.info(f"Prioritized to {len(tools_to_process)} tools within OpenAI limits")
                
                # Log which tools were selected
                selected_names = [tool["name"] for tool in tools_to_process]
                logger.debug(f"Selected tools: {selected_names[:10]}...")  # Log first 10
                
                # Log which GA4 tools are available and whether they were selected
                all_tool_names = [tool["name"] for tool in raw_tools]
                ga4_tools_names = [name for name in all_tool_names if "ga4" in name.lower()]
                selected_ga4_tools = [name for name in selected_names if "ga4" in name.lower()]
                logger.info(f"Available GA4 tools in server response: {ga4_tools_names}")
                logger.info(f"Selected GA4 tools after prioritization: {selected_ga4_tools}")
                
                # Log which breeze tools are available
                breeze_tools = [name for name in all_tool_names if "breeze" in name.lower()]
                selected_breeze_tools = [name for name in selected_names if "breeze" in name.lower()]
                logger.info(f"Available breeze tools: {len(breeze_tools)} total")
                logger.info(f"Selected breeze tools: {len(selected_breeze_tools)} selected")
            
            converted_tools = []
            for tool_data in tools_to_process:
                tool_name = tool_data["name"]
                logger.debug(f"Registering remote tool: {tool_name}")
                
                # Debug schema for specific problematic tools
                if "getBusinessAnalyticsCounts" in tool_name:
                    logger.info(f"DEBUG: Schema for {tool_name}:")
                    logger.info(f"  Raw tool data: {json.dumps(tool_data, indent=2)}")
                
                function_schema = self._convert_schema(tool_data)
                converted_tools.append(function_schema)
                
                # Log the converted schema for problematic tools
                if "getBusinessAnalyticsCounts" in tool_name:
                    logger.info(f"  Converted function schema: name={function_schema.name}")
                    logger.info(f"  Properties: {function_schema.properties}")
                    logger.info(f"  Required: {function_schema.required}")
                
                # Register using the potentially shortened name from the schema
                llm.register_function(function_schema.name, self._mcp_tool_wrapper)
                
            logger.info(f"Successfully registered {len(converted_tools)} remote tools.")
            return ToolsSchema(standard_tools=converted_tools)
        except Exception as e:
            logger.error(f"Failed to register tools from remote server: {e}")
            return ToolsSchema(standard_tools=[])

    def _convert_schema(self, tool_data: Dict[str, Any]) -> FunctionSchema:
        """Converts a raw MCP tool dict to a PipeCat FunctionSchema."""
        tool = MCPTool.model_validate(tool_data)
        
        # OpenAI has a 64-character limit on function names
        original_name = tool.name
        if len(original_name) > 64:
            # Create a shorter name by truncating and adding a hash suffix
            import hashlib
            hash_suffix = hashlib.md5(original_name.encode()).hexdigest()[:8]
            truncated_name = original_name[:50] + "_" + hash_suffix  # 50 + 1 + 8 = 59 chars
            
            logger.info(f"Tool name too long ({len(original_name)} chars): '{original_name}' -> '{truncated_name}'")
            # Store the mapping for tool calls
            self._name_mapping = getattr(self, '_name_mapping', {})
            self._name_mapping[truncated_name] = original_name
            function_name = truncated_name
        else:
            function_name = original_name
        
        return FunctionSchema(
            name=function_name,
            description=tool.description,
            properties=tool.input_schema.properties,
            required=tool.input_schema.required or [],
        )

    async def _mcp_tool_wrapper(self, params) -> None:
        """This wrapper is called by the LLM. It then calls the remote tool."""
        # Extract parameters from the params object
        function_name = getattr(params, 'function_name', 'unknown')
        arguments = getattr(params, 'arguments', {})
        result_callback = getattr(params, 'result_callback', None)
        
        # Check if this is a truncated name that needs to be mapped back
        name_mapping = getattr(self, '_name_mapping', {})
        actual_function_name = name_mapping.get(function_name, function_name)
        
        if function_name != actual_function_name:
            logger.debug(f"Mapped truncated name '{function_name}' back to original '{actual_function_name}'")
        
        logger.debug(f"LLM called MCP tool: {actual_function_name} with args: {arguments}")
        
        if not result_callback:
            logger.error(f"No result_callback found for MCP function {function_name}")
            return
            
        await self._call_tool(actual_function_name, arguments, result_callback)

    async def _call_tool(
        self, function_name: str, arguments: Dict[str, Any], result_callback: Callable
    ) -> None:
        """Sends the 'tools/call' request to the remote server."""
        try:
            # Debug for problematic tools
            if "getBusinessAnalyticsCounts" in function_name:
                logger.info(f"DEBUG: Tool call for {function_name}")
                logger.info(f"  LLM sent arguments: {json.dumps(arguments, indent=2)}")
                logger.info(f"  Context merchantId: {self._transport._context.get('merchantId')}")
                
                # If this is a timeframe-based call, we need to convert it to date range
                if 'timeframe' in arguments and 'startDate' not in arguments:
                    logger.warning(f"Tool {function_name} received timeframe '{arguments['timeframe']}' but server expects startDate/endDate")
                    
                    # Convert timeframe to actual dates
                    from datetime import datetime, timedelta
                    now = datetime.now()
                    
                    timeframe = arguments['timeframe']
                    if timeframe == 'this_week':
                        # Start of this week (Monday)
                        days_since_monday = now.weekday()
                        start_date = now - timedelta(days=days_since_monday)
                        end_date = now
                    elif timeframe == 'last_week':
                        # Start and end of last week
                        days_since_monday = now.weekday()
                        this_monday = now - timedelta(days=days_since_monday)
                        start_date = this_monday - timedelta(days=7)
                        end_date = this_monday - timedelta(days=1)
                    elif timeframe == 'this_month':
                        # Start of this month
                        start_date = now.replace(day=1)
                        end_date = now
                    elif timeframe == 'last_month':
                        # Start and end of last month
                        first_day_this_month = now.replace(day=1)
                        end_date = first_day_this_month - timedelta(days=1)
                        start_date = end_date.replace(day=1)
                    else:
                        # Default to this week
                        days_since_monday = now.weekday()
                        start_date = now - timedelta(days=days_since_monday)
                        end_date = now
                    
                    # Remove timeframe and add startDate/endDate
                    del arguments['timeframe']
                    arguments['startDate'] = start_date.strftime('%Y-%m-%d')
                    arguments['endDate'] = end_date.strftime('%Y-%m-%d')
                    logger.info(f"  Converted timeframe '{timeframe}' to startDate: {arguments['startDate']}, endDate: {arguments['endDate']}")
                    
                # If merchantIds is missing but we have merchantId in context, add it
                if 'merchantIds' not in arguments and self._transport._context.get('merchantId'):
                    arguments['merchantIds'] = [self._transport._context['merchantId']]
                    logger.info(f"  Added merchantIds from context: {arguments['merchantIds']}")
            
            params = {"name": function_name, "arguments": arguments}
            logger.info(f"Calling MCP tool '{function_name}' with arguments: {arguments}")
            logger.debug(f"Tool call params: {params}")
            
            # The transport.post method will automatically include x-context and x-auth-token headers
            response_dict = await self._transport.post(method="tools/call", params=params)
            
            logger.debug(f"Tool call response received for '{function_name}': {response_dict}")

            if response_dict.get("error"):
                error_details = response_dict['error']
                logger.error(f"Tool call error for '{function_name}': {error_details}")
                raise RuntimeError(f"JSON-RPC Error calling tool: {error_details}")

            result_content = response_dict.get("result", {}).get("content", [])
            
            text_response = " ".join(
                json.dumps(item.get("text")) for item in result_content if item.get("type") == "text"
            )

            if not text_response:
                text_response = "Tool executed successfully but returned no text."

            logger.debug(f"Tool '{function_name}' returned: {text_response}")
            await result_callback(text_response)

        except Exception as e:
            logger.error(f"Failed to call tool '{function_name}': {e}")
            await result_callback(f"Error: Could not execute tool {function_name}.")

    async def close(self):
        await self._client.aclose()
