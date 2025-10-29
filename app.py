#!/usr/bin/env python3
"""
Appalachian Map Chatbot - Flask Backend with OpenAI Integration
"""
from flask import Flask, render_template, request, jsonify, send_file, make_response
import pandas as pd
import json
import re
import os
import numpy as np
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from io import BytesIO
from datetime import datetime

# Load environment variables
try:
    load_dotenv()
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")
    print("Please ensure your .env file is properly formatted with UTF-8 encoding")

app = Flask(__name__)

# Initialize OpenAI client - handle missing API key gracefully
try:
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key and api_key != 'your_openai_api_key_here':
        openai_client = OpenAI(api_key=api_key)
    else:
        print("WARNING: OpenAI API key not set. Advanced NLP features will be disabled.")
        openai_client = None
except Exception as e:
    print(f"WARNING: Could not initialize OpenAI client: {e}")
    openai_client = None

# Load data
def load_data():
    """Load all necessary data files"""
    data = {}
    
    # Load counties master list
    data['counties'] = pd.read_csv('data_clean/counties.csv', dtype={'fips': str})
    
    # Load unemployment data
    data['unemployment'] = pd.read_csv('data_clean/unemployment_normalized.csv', dtype={'fips': str})
    
    # Load GDP data
    data['gdp'] = pd.read_csv('data_clean/gdp_with_per_capita.csv', dtype={'fips': str})
    
    # Load county summary
    data['summary'] = pd.read_csv('data_clean/county_summary.csv', dtype={'fips': str})
    
    # Load income limits data
    try:
        data['income_limits'] = pd.read_csv('data_clean/income_limits.csv', dtype={'fips': str})
        # Extract FIPS code (first 5 digits)
        data['income_limits']['fips'] = data['income_limits']['fips'].astype(str).str[:5].str.zfill(5)
        print(f"Loaded income limits data for {len(data['income_limits'])} records")
    except Exception as e:
        print(f"Warning: Could not load income limits data: {e}")
        data['income_limits'] = pd.DataFrame()
    
    # Load statewide GDP data for comparison
    statewide_file = 'data_statewide/statewide_gdp_per_capita.csv'
    if os.path.exists(statewide_file):
        data['statewide_gdp'] = pd.read_csv(statewide_file)
    else:
        print(f"Warning: {statewide_file} not found. Creating default data...")
        data['statewide_gdp'] = pd.DataFrame({
            'state': ['Alabama', 'Georgia', 'Kentucky', 'Maryland', 'Mississippi', 
                     'New York', 'North Carolina', 'Ohio', 'Pennsylvania', 
                     'South Carolina', 'Tennessee', 'Virginia', 'West Virginia'],
            'gdp_per_capita': [44500, 52000, 47500, 62000, 38000, 75000, 
                              56000, 53000, 59000, 48000, 51000, 61000, 42000]
        })
    
    # Load TopoJSON - much smaller file size (630KB vs 30MB)
    with open('data_clean/appalachia_snapshot.topojson', 'r') as f:
        data['topojson'] = json.load(f)
    
    # Pre-process GeoJSON for faster rendering (simplify if needed)
    # Note: TopoJSON is already simplified and compressed
    print(f"Loaded TopoJSON with arcs and features")
    
    # Also keep GeoJSON for backwards compatibility if needed
    with open('data_clean/appalachia_snapshot.geojson', 'r') as f:
        data['geojson'] = json.load(f)
    print(f"Loaded GeoJSON with {len(data['geojson']['features'])} features")
    
    return data

# Load data at startup
data = load_data()

class AppalachianChatbot:
    """Natural language processing for Appalachian county queries with OpenAI integration"""
    
    def __init__(self, data):
        self.data = data
        self.counties = data['counties']
        self.unemployment = data['unemployment']
        self.gdp = data['gdp']
        self.summary = data['summary']
        self.income_limits = data.get('income_limits', pd.DataFrame())
        self.statewide_gdp = data.get('statewide_gdp', None)
        self.openai_client = openai_client
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            val = float(obj)
            # Convert NaN to None (null in JSON)
            if np.isnan(val):
                return None
            return val
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        # Handle pandas NA/NaN values
        if pd.isna(obj) or obj is pd.NA:
            return None
        return obj
    
    def parse_query_with_openai(self, query: str) -> Dict[str, Any]:
        """Use OpenAI to parse natural language query and extract parameters"""
        # Fallback to regex if OpenAI client is not available
        if not self.openai_client:
            return self.parse_query_regex(query)
        
        try:
            # Create a system prompt that explains the data structure
            system_prompt = """
            You are an intelligent data analyst assistant for Appalachian county economic data. 
            Your job is to understand user queries and extract structured information for data analysis.
            
            Available data fields:
            - unemployment_rate (percentage, 0-100)
            - gdp_per_capita (dollars, typically 5,000-100,000)
            - state (Alabama, Georgia, Kentucky, Maryland, Mississippi, New York, North Carolina, Ohio, Pennsylvania, South Carolina, Tennessee, Virginia, West Virginia)
            - county (county name)
            
            Parse queries and return JSON with this structure:
            {
                "filters": {
                    "unemployment_min": null,
                    "unemployment_max": null,
                    "gdp_per_capita_min": null,
                    "gdp_per_capita_max": null,
                    "state": null,
                    "county": null
                },
                "query_type": "list|info|count|analytics|greeting|conversation",
                "limit": null
            }
            
            Query types:
            - "list": Show/find/list counties (data queries)
            - "info": What/how questions about specific data
            - "count": How many questions
            - "analytics": Statistical analysis (average, median, max, min, comparison)
            - "greeting": Hello, hi, help, etc.
            - "conversation": General conversation not requiring data
            
            CRITICAL: Use exact filter names. For "greater than" or ">", use "unemployment_min". For "less than" or "<", use "unemployment_max".
            
            Examples:
            - "Show counties with unemployment rate > 7%" -> {"filters": {"unemployment_min": 7}, "query_type": "list"}
            - "Show counties with unemployment rate rate > 8%" -> {"filters": {"unemployment_min": 8}, "query_type": "list"}
            - "Find counties in Alabama" -> {"filters": {"state": "Alabama"}, "query_type": "list"}
            - "Counties with GDP per capita > $55,000" -> {"filters": {"gdp_per_capita_min": 55000}, "query_type": "list"}
            - "List counties with GDP per capita > $55,000 in Alabama" -> {"filters": {"gdp_per_capita_min": 55000, "state": "Alabama"}, "query_type": "list"}
            - "What is the average GDP per capita in Virginia?" -> {"filters": {"state": "Virginia"}, "query_type": "analytics"}
            - "What's the highest unemployment rate in Kentucky?" -> {"filters": {"state": "Kentucky"}, "query_type": "analytics"}
            - "What's the unemployment rate in Calhoun County?" -> {"filters": {"county": "Calhoun"}, "query_type": "info"}
            - "What's the unemployment rate in Calhoun County, West Virginia?" -> {"filters": {"county": "Calhoun", "state": "West Virginia"}, "query_type": "info"}
            - "Compare unemployment rates between Alabama and Georgia" -> {"filters": {}, "query_type": "analytics"}
            - "hello" -> {"filters": {}, "query_type": "greeting"}
            - "help" -> {"filters": {}, "query_type": "greeting"}
            - "tell me about yourself" -> {"filters": {}, "query_type": "conversation"}
            
            IMPORTANT: Always return valid JSON. Do not include any text outside the JSON structure.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Parse this query: {query}"}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # Extract JSON from response
            content = response.choices[0].message.content.strip()
            
            # Try to extract JSON from markdown code blocks if present
            import re as re_module
            json_match = re_module.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re_module.DOTALL)
            if json_match:
                content = json_match.group(1)
            
            # Try to parse JSON
            try:
                result = json.loads(content)
                # Ensure required fields exist
                if 'filters' not in result:
                    result['filters'] = {}
                if 'query_type' not in result:
                    result['query_type'] = 'general'
                print(f"Parsed query: {result}")  # Debug logging
                return result
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}, content: {content}")
                # Fallback to regex parsing if OpenAI response is not valid JSON
                return self.parse_query_regex(query)
                
        except Exception as e:
            print(f"OpenAI parsing failed: {e}")
            # Fallback to regex parsing
            return self.parse_query_regex(query)
    
    def parse_query_regex(self, query: str) -> Dict[str, Any]:
        """Fallback regex-based query parsing"""
        query = query.lower().strip()
        
        # Handle simple greetings and non-data queries
        greeting_patterns = [
            r'^(hello|hi|hey|greetings|good morning|good afternoon|good evening)$',
            r'^(how are you|what can you do|help|what do you do)$',
            r'^(thanks|thank you|bye|goodbye)$'
        ]
        
        for pattern in greeting_patterns:
            if re.match(pattern, query):
                return {
                    'filters': {},
                    'query_type': 'greeting',
                    'limit': None
                }
        
        # Initialize result
        result = {
            'filters': {},
            'sort_by': None,
            'limit': None,
            'query_type': 'general'
        }
        
        # Extract unemployment rate filters
        unemployment_patterns = [
            r'unemployment\s*(?:rate)?\s*(?:>|greater than|above|over)\s*(\d+(?:\.\d+)?)',
            r'unemployment\s*(?:rate)?\s*(?:<|less than|below|under)\s*(\d+(?:\.\d+)?)',
            r'unemployment\s*(?:rate)?\s*(?:=|=|is)\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)%?\s*(?:unemployment|unemployed)'
        ]
        
        for i, pattern in enumerate(unemployment_patterns):
            match = re.search(pattern, query)
            if match:
                value = float(match.group(1))
                if i == 0:  # First pattern has '>' or 'greater than' etc
                    result['filters']['unemployment_min'] = value
                elif i == 1:  # Second pattern has '<' or 'less than' etc
                    result['filters']['unemployment_max'] = value
                else:  # Other patterns are exact matches
                    result['filters']['unemployment_exact'] = value
                break
        
        # Extract GDP per capita filters
        gdp_patterns = [
            r'gdp\s*(?:per capita)?\s*(?:>|greater than|above|over)\s*\$?(\d+(?:,\d+)?(?:\.\d+)?)',
            r'gdp\s*(?:per capita)?\s*(?:<|less than|below|under)\s*\$?(\d+(?:,\d+)?(?:\.\d+)?)',
            r'gdp\s*(?:per capita)?\s*(?:=|=|is)\s*\$?(\d+(?:,\d+)?(?:\.\d+)?)',
            r'\$(\d+(?:,\d+)?(?:\.\d+)?)\s*(?:gdp|per capita)'
        ]
        
        for i, pattern in enumerate(gdp_patterns):
            match = re.search(pattern, query)
            if match:
                value = float(match.group(1).replace(',', ''))
                if i == 0:  # First pattern has '>' or 'greater than' etc
                    result['filters']['gdp_per_capita_min'] = value
                elif i == 1:  # Second pattern has '<' or 'less than' etc
                    result['filters']['gdp_per_capita_max'] = value
                else:  # Other patterns are exact matches
                    result['filters']['gdp_per_capita_exact'] = value
                break
        
        # Extract state filters
        state_patterns = [
            r'in\s+([a-z\s]+)',
            r'from\s+([a-z\s]+)',
            r'([a-z\s]+)\s+counties?'
        ]
        
        for pattern in state_patterns:
            match = re.search(pattern, query)
            if match:
                state_name = match.group(1).strip()
                # Map common state names
                state_mapping = {
                    'alabama': 'Alabama',
                    'georgia': 'Georgia', 
                    'kentucky': 'Kentucky',
                    'maryland': 'Maryland',
                    'mississippi': 'Mississippi',
                    'new york': 'New York',
                    'north carolina': 'North Carolina',
                    'ohio': 'Ohio',
                    'pennsylvania': 'Pennsylvania',
                    'south carolina': 'South Carolina',
                    'tennessee': 'Tennessee',
                    'virginia': 'Virginia',
                    'west virginia': 'West Virginia'
                }
                if state_name in state_mapping:
                    result['filters']['state'] = state_mapping[state_name]
                break
        
        # Extract county name filters
        county_patterns = [
            r'in\s+(\w+)\s+county',
            r'(\w+)\s+county',
            r'county\s+(\w+)',
            r'in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:county|County)'  # Multi-word county names
        ]
        
        for pattern in county_patterns:
            match = re.search(pattern, query)
            if match:
                county_name = match.group(1).strip().title()
                result['filters']['county'] = county_name
                break
        
        # Determine query type
        if 'show' in query or 'list' in query or 'find' in query:
            result['query_type'] = 'list'
        elif 'what' in query or 'how' in query or 'rate' in query:
            result['query_type'] = 'info'
        elif 'count' in query or 'how many' in query:
            result['query_type'] = 'count'
        
        return result
    
    def execute_query(self, parsed_query: Dict[str, Any]) -> pd.DataFrame:
        """Execute the parsed query and return results"""
        # Handle greeting queries
        if parsed_query.get('query_type') == 'greeting':
            return pd.DataFrame()  # Empty DataFrame for greetings
        
        print(f"Execute query with filters: {parsed_query.get('filters', {})}")
        
        # Start with summary data
        results = self.summary.copy()
        print(f"Starting with {len(results)} counties")
        
        # Apply filters
        if 'unemployment_min' in parsed_query['filters']:
            threshold = parsed_query['filters']['unemployment_min']
            print(f"Filtering for unemployment_rate >= {threshold}")
            results = results[results['unemployment_rate'] >= threshold]
            print(f"After filter: {len(results)} counties")
        
        if 'unemployment_max' in parsed_query['filters']:
            results = results[results['unemployment_rate'] <= parsed_query['filters']['unemployment_max']]
        
        if 'unemployment_exact' in parsed_query['filters']:
            results = results[abs(results['unemployment_rate'] - parsed_query['filters']['unemployment_exact']) < 0.5]
        
        if 'gdp_per_capita_min' in parsed_query['filters']:
            results = results[results['gdp_per_capita'] >= parsed_query['filters']['gdp_per_capita_min']]
        
        if 'gdp_per_capita_max' in parsed_query['filters']:
            results = results[results['gdp_per_capita'] <= parsed_query['filters']['gdp_per_capita_max']]
        
        if 'gdp_per_capita_exact' in parsed_query['filters']:
            results = results[abs(results['gdp_per_capita'] - parsed_query['filters']['gdp_per_capita_exact']) < 1000]
        
        if 'state' in parsed_query['filters']:
            results = results[results['state'] == parsed_query['filters']['state']]
        
        if 'county' in parsed_query['filters'] and parsed_query['filters']['county']:
            county_value = str(parsed_query['filters']['county'])
            # Remove "County" suffix if present in the search term
            county_value = county_value.replace(' County', '').replace(' county', '').strip()
            results = results[results['county'].str.contains(county_value, case=False, na=False)]
        
        # Remove rows with missing data
        results = results.dropna(subset=['unemployment_rate', 'gdp_per_capita'])
        
        return results
    
    def perform_analytics(self, parsed_query: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """Perform statistical analysis on the data"""
        # Check if this is a statewide comparison query
        is_statewide_comparison = self._is_statewide_comparison_query(original_query)
        
        # Get filtered data
        results = self.execute_query(parsed_query)
        
        if len(results) == 0:
            return {
                'success': False,
                'message': 'No data available for the specified criteria to perform analysis.',
                'analytics': {}
            }
        
        # Perform various analytical calculations
        analytics = {}
        
        # Basic statistics
        analytics['count'] = len(results)
        analytics['avg_unemployment'] = results['unemployment_rate'].mean()
        analytics['avg_gdp_per_capita'] = results['gdp_per_capita'].mean()
        
        # Min/Max values
        analytics['min_unemployment'] = results['unemployment_rate'].min()
        analytics['max_unemployment'] = results['unemployment_rate'].max()
        analytics['min_gdp_per_capita'] = results['gdp_per_capita'].min()
        analytics['max_gdp_per_capita'] = results['gdp_per_capita'].max()
        
        # Median values
        analytics['median_unemployment'] = results['unemployment_rate'].median()
        analytics['median_gdp_per_capita'] = results['gdp_per_capita'].median()
        
        # Standard deviation
        analytics['std_unemployment'] = results['unemployment_rate'].std()
        analytics['std_gdp_per_capita'] = results['gdp_per_capita'].std()
        
        # Add statewide comparison if requested
        if is_statewide_comparison and self.statewide_gdp is not None:
            state = parsed_query['filters'].get('state')
            if state:
                statewide_value = self.statewide_gdp[self.statewide_gdp['state'] == state]
                if len(statewide_value) > 0:
                    analytics['statewide_gdp_per_capita'] = statewide_value.iloc[0]['gdp_per_capita']
                    analytics['is_statewide_comparison'] = True
        
        # State breakdown if multiple states
        if 'state' not in parsed_query['filters'] or parsed_query['filters']['state'] is None:
            state_stats = results.groupby('state').agg({
                'unemployment_rate': ['mean', 'min', 'max', 'count'],
                'gdp_per_capita': ['mean', 'min', 'max']
            }).round(2)
            analytics['state_breakdown'] = state_stats.to_dict()
        
        # Generate intelligent analysis using OpenAI
        analysis_message = self._generate_analytics_message(original_query, analytics, results)
        
        # Convert analytics to native Python types for JSON serialization
        analytics_converted = self._convert_numpy_types(analytics)
        counties_list = results.to_dict('records')
        counties_list = self._convert_numpy_types(counties_list)
        
        return {
            'success': True,
            'message': analysis_message,
            'analytics': analytics_converted,
            'counties': counties_list,
            'count': len(results)
        }
    
    def _is_statewide_comparison_query(self, query: str) -> bool:
        """Check if the query is asking for statewide comparison"""
        query_lower = query.lower()
        statewide_keywords = [
            'statewide', 'state average', 'entire state', 'all of', 
            'state as a whole', 'whole state', 'state-wide', 'average gdp per capita of',
            'average of al', 'average of va', 'average of'
        ]
        # Also check for pattern like "average GDP per capita of AL" or "average of AL"
        state_patterns = ['average.*of al', 'average.*of va', 'average.*of ga', 'average.*of ky']
        return any(keyword in query_lower for keyword in statewide_keywords) or \
               any(re.search(pattern, query_lower) for pattern in state_patterns)
    
    def _generate_analytics_message(self, original_query: str, analytics: Dict[str, Any], results: pd.DataFrame) -> str:
        """Generate intelligent analytical insights using OpenAI"""
        if not self.openai_client:
            return self._generate_simple_analytics_message(analytics, results)
        
        try:
            # Find specific counties for min/max values
            min_gdp_county = results.loc[results['gdp_per_capita'].idxmin()]
            max_gdp_county = results.loc[results['gdp_per_capita'].idxmax()]
            min_unemp_county = results.loc[results['unemployment_rate'].idxmin()]
            max_unemp_county = results.loc[results['unemployment_rate'].idxmax()]
            
            # Create context for analysis
            context = f"""
            User asked: "{original_query}"
            
            Analysis Results:
            - Number of counties analyzed: {analytics['count']}
            - Average unemployment rate: {analytics['avg_unemployment']:.2f}%
            - Average GDP per capita: ${analytics['avg_gdp_per_capita']:.0f}
            - Unemployment range: {analytics['min_unemployment']:.1f}% to {analytics['max_unemployment']:.1f}%
            - GDP per capita range: ${analytics['min_gdp_per_capita']:.0f} to ${analytics['max_gdp_per_capita']:.0f}
            - Median unemployment: {analytics['median_unemployment']:.2f}%
            - Median GDP per capita: ${analytics['median_gdp_per_capita']:.0f}
            
            Specific Counties:
            - Lowest GDP per capita: {min_gdp_county['county']}, {min_gdp_county['state']} (${min_gdp_county['gdp_per_capita']:.0f})
            - Highest GDP per capita: {max_gdp_county['county']}, {max_gdp_county['state']} (${max_gdp_county['gdp_per_capita']:.0f})
            - Lowest unemployment: {min_unemp_county['county']}, {min_unemp_county['state']} ({min_unemp_county['unemployment_rate']:.1f}%)
            - Highest unemployment: {max_unemp_county['county']}, {max_unemp_county['state']} ({max_unemp_county['unemployment_rate']:.1f}%)
            """
            
            # Add statewide comparison if available
            if 'statewide_gdp_per_capita' in analytics:
                context += f"\n\nStatewide Comparison:\n- Statewide GDP per capita: ${analytics['statewide_gdp_per_capita']:.0f}\n- Appalachian counties GDP per capita: ${analytics['avg_gdp_per_capita']:.0f}\n- Difference: ${analytics['statewide_gdp_per_capita'] - analytics['avg_gdp_per_capita']:.0f}"
            
            # Add state breakdown if available
            if 'state_breakdown' in analytics:
                context += "\n\nState breakdown available for comparison."
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data analyst expert. Provide SHORT, DIRECT answers in bullet point format. Answer the user's specific question first, then provide relevant supporting data. Be concise and to the point. Use bullet points for clarity. When comparing statewide vs Appalachian data, highlight the key differences."},
                    {"role": "user", "content": f"Answer this question concisely with bullet points:\n{context}"}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"OpenAI analytics generation failed: {e}")
            # Fallback to simple analysis
            return self._generate_simple_analytics_message(analytics, results)
    
    def _generate_simple_analytics_message(self, analytics: Dict[str, Any], results: pd.DataFrame) -> str:
        """Fallback simple analytics message"""
        # Find specific counties for min/max values
        min_gdp_county = results.loc[results['gdp_per_capita'].idxmin()]
        max_gdp_county = results.loc[results['gdp_per_capita'].idxmax()]
        min_unemp_county = results.loc[results['unemployment_rate'].idxmin()]
        max_unemp_county = results.loc[results['unemployment_rate'].idxmax()]
        
        return f"""• Counties analyzed: {analytics['count']}

• **Unemployment Rate:**
  - Average: {analytics['avg_unemployment']:.2f}%
  - Range: {analytics['min_unemployment']:.1f}% to {analytics['max_unemployment']:.1f}%
  - Lowest: {min_unemp_county['county']}, {min_unemp_county['state']} ({min_unemp_county['unemployment_rate']:.1f}%)
  - Highest: {max_unemp_county['county']}, {max_unemp_county['state']} ({max_unemp_county['unemployment_rate']:.1f}%)

• **GDP per Capita:**
  - Average: ${analytics['avg_gdp_per_capita']:.0f}
  - Range: ${analytics['min_gdp_per_capita']:.0f} to ${analytics['max_gdp_per_capita']:.0f}
  - Lowest: {min_gdp_county['county']}, {min_gdp_county['state']} (${min_gdp_county['gdp_per_capita']:.0f})
  - Highest: {max_gdp_county['county']}, {max_gdp_county['state']} (${max_gdp_county['gdp_per_capita']:.0f})"""
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """Generate a response to a natural language query using OpenAI"""
        # Try OpenAI parsing first, fallback to regex
        parsed_query = self.parse_query_with_openai(query)
        
        # Handle greeting queries specially
        if parsed_query.get('query_type') == 'greeting':
            return self._handle_greeting_query(query)
        
        # Handle conversation queries
        if parsed_query.get('query_type') == 'conversation':
            return self._handle_conversation_query(query)
        
        # Handle analytics queries
        if parsed_query.get('query_type') == 'analytics':
            return self.perform_analytics(parsed_query, query)
        
        # Handle data queries
        results = self.execute_query(parsed_query)
        
        if len(results) == 0:
            return {
                'success': False,
                'message': 'No counties found matching your criteria. Try adjusting your search parameters or ask me for help with different criteria.',
                'counties': []
            }
        
        # Generate response message using OpenAI
        message = self._generate_openai_message(query, parsed_query, results)
        
        # Convert results to list of dictionaries and convert numpy types
        counties_list = results.to_dict('records')
        counties_list = self._convert_numpy_types(counties_list)
        
        return {
            'success': True,
            'message': message,
            'counties': counties_list,
            'count': len(counties_list)
        }
    
    def _handle_greeting_query(self, query: str) -> Dict[str, Any]:
        """Handle greeting queries with intelligent responses"""
        greeting_responses = {
            'hello': "Hello! I'm your intelligent Appalachian counties data assistant. I can help you explore economic data like unemployment rates and GDP per capita across 425 Appalachian counties. Try asking me something like 'Show counties with unemployment rate > 8%' or 'Find counties in Alabama with high GDP per capita'.",
            'hi': "Hi there! I'm here to help you analyze Appalachian county data. I can answer questions about unemployment, GDP, specific states, or counties. What would you like to explore?",
            'hey': "Hey! I'm your data-savvy assistant for Appalachian counties. I can help you find counties with specific economic conditions, compare states, or answer questions about local economies. What interests you?",
            'help': "I can help you explore Appalachian county data! Here are some things you can ask me:\n• 'Show counties with unemployment rate > 8%'\n• 'Find counties with GDP per capita > $55,000'\n• 'List counties in Alabama'\n• 'What's the unemployment rate in Calhoun County?'\n• 'What is the average GDP per capita in Virginia?'\n• 'Compare unemployment rates between Alabama and Georgia'\n• 'What's the highest unemployment rate in Kentucky?'\n• 'Tell me about the economic conditions in West Virginia'",
            'what can you do': "I'm an intelligent data analyst for Appalachian counties! I can:\n• Analyze unemployment rates across 425 counties\n• Compare GDP per capita data\n• Filter counties by state or economic conditions\n• Calculate averages, medians, min/max values\n• Compare states and regions\n• Answer complex questions combining multiple criteria\n• Provide statistical insights and analysis\n\nTry asking me analytical questions like 'What is the average GDP per capita in Virginia?' or 'Compare unemployment rates between states'!"
        }
        
        query_lower = query.lower().strip()
        for greeting, response in greeting_responses.items():
            if greeting in query_lower:
                return {
                    'success': True,
                    'message': response,
                    'counties': [],
                    'count': 0
                }
        
        # Default greeting response
        return {
            'success': True,
            'message': "Hello! I'm your intelligent Appalachian counties data assistant. I can help you explore economic data across 425 Appalachian counties. Try asking me about unemployment rates, GDP per capita, specific states, or counties!",
            'counties': [],
            'count': 0
        }
    
    def _handle_conversation_query(self, query: str) -> Dict[str, Any]:
        """Handle general conversation queries using OpenAI"""
        if not self.openai_client:
            return {
                'success': True,
                'message': "I'm here to help you explore Appalachian county data! I can analyze unemployment rates, GDP per capita, and help you find counties with specific economic conditions. What would you like to know?",
                'counties': [],
                'count': 0
            }
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst assistant specializing in Appalachian county economic data. You can discuss economics, geography, data analysis, and help users understand Appalachian counties. Keep responses conversational but informative."},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            return {
                'success': True,
                'message': response.choices[0].message.content.strip(),
                'counties': [],
                'count': 0
            }
            
        except Exception as e:
            print(f"OpenAI conversation failed: {e}")
            return {
                'success': True,
                'message': "I'm here to help you explore Appalachian county data! I can analyze unemployment rates, GDP per capita, and help you find counties with specific economic conditions. What would you like to know?",
                'counties': [],
                'count': 0
            }
    
    def _generate_openai_message(self, original_query: str, parsed_query: Dict[str, Any], results: pd.DataFrame) -> str:
        """Generate a human-readable response message using OpenAI"""
        if not self.openai_client:
            return self._generate_simple_message(parsed_query, results)
        
        try:
            count = len(results)
            
            # Create context about the results
            context = f"""
            User asked: "{original_query}"
            Found {count} counties matching the criteria.
            
            Sample results:
            """
            
            # Add sample results with more detail
            if count > 0:
                # For queries about specific counties (e.g., "What's the unemployment rate in Calhoun County?"), 
                # include all results if there are multiple counties with the same name
                if count <= 5:
                    # Show all results if 5 or fewer
                    for i in range(count):
                        county = results.iloc[i]
                        context += f"- {county['county']}, {county['state']}: Unemployment {county['unemployment_rate']:.1f}%, GDP per capita ${county['gdp_per_capita']:.0f}\n"
                else:
                    # Show first 5 results if more than 5
                    sample_size = 5
                    for i in range(sample_size):
                        county = results.iloc[i]
                        context += f"- {county['county']}, {county['state']}: Unemployment {county['unemployment_rate']:.1f}%, GDP per capita ${county['gdp_per_capita']:.0f}\n"
                
                # Add summary statistics
                avg_unemployment = results['unemployment_rate'].mean()
                avg_gdp = results['gdp_per_capita'].mean()
                context += f"\nSummary: Average unemployment {avg_unemployment:.1f}%, Average GDP per capita ${avg_gdp:.0f}"
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst assistant. Provide SHORT, DIRECT responses about Appalachian county data. Use bullet points for clarity. Be concise and to the point."},
                    {"role": "user", "content": f"Generate a brief, informative response for this query result:\n{context}"}
                ],
                temperature=0.2,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"OpenAI message generation failed: {e}")
            # Fallback to simple message
            return self._generate_simple_message(parsed_query, results)
    
    def _generate_simple_message(self, parsed_query: Dict[str, Any], results: pd.DataFrame) -> str:
        """Fallback simple message generation"""
        count = len(results)
        
        if parsed_query['query_type'] == 'count':
            return f"Found {count} counties matching your criteria."
        
        if count == 1:
            county = results.iloc[0]
            return f"Found 1 county: {county['county']}, {county['state']}"
        
        if count <= 10:
            return f"Found {count} counties matching your criteria."
        else:
            return f"Found {count} counties matching your criteria. Showing all results."

# Initialize chatbot
chatbot = AppalachianChatbot(data)

@app.route('/public/<path:filename>')
def public_file(filename):
    """Serve static files from the public directory"""
    try:
        import mimetypes
        file_path = os.path.join('public', filename)
        if not os.path.exists(file_path):
            return f"File not found: {filename}", 404
        mimetype, _ = mimetypes.guess_type(filename)
        if not mimetype:
            mimetype = 'application/octet-stream'
        return send_file(file_path, mimetype=mimetype)
    except Exception as e:
        print(f"Error serving file {filename}: {e}")
        return f"Error serving file: {e}", 500

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/counties')
def get_counties():
    """Return GeoJSON data for counties"""
    response = jsonify(data['geojson'])
    # Add cache headers for better performance
    response.headers['Cache-Control'] = 'public, max-age=3600'
    return response

@app.route('/api/counties/topojson')
def get_counties_topojson():
    """Return TopoJSON data for counties (smaller file size)"""
    response = jsonify(data['topojson'])
    # Add cache headers for better performance
    response.headers['Cache-Control'] = 'public, max-age=3600'
    return response

@app.route('/api/summary')
def get_summary():
    """Return summary statistics for counties"""
    chatbot = AppalachianChatbot(data)
    summary_records = data['summary'].to_dict('records')
    summary_converted = chatbot._convert_numpy_types(summary_records)
    return jsonify(summary_converted)

@app.route('/api/query', methods=['POST'])
def process_query():
    """Process natural language queries"""
    try:
        query_data = request.get_json()
        query = query_data.get('query', '')
        
        if not query:
            return jsonify({
                'success': False,
                'message': 'Please provide a query.'
            })
        
        response = chatbot.generate_response(query)
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error processing query: {str(e)}'
        })

@app.route('/api/data/<data_type>')
def get_data(data_type):
    """Return specific data types"""
    if data_type in data:
        return jsonify(data[data_type].to_dict('records'))
    else:
        return jsonify({'error': 'Data type not found'}), 404

@app.route('/api/income_limits')
def get_income_limits():
    """Return income limits data with optional family size and income type filter"""
    try:
        family_size = request.args.get('family_size', '1')  # Default to 1 person
        income_type = request.args.get('income_type', 'very_low')  # Default to very_low
        family_size = int(family_size)
        
        if family_size < 1 or family_size > 8:
            return jsonify({'error': 'Family size must be between 1 and 8'}), 400
        
        # Handle median income (doesn't need family size)
        if income_type == 'median':
            column_name = 'median2025'
        else:
            # Get the appropriate column for the family size and income type
            column_name = f'{income_type}_{family_size}_person'
        
        if column_name not in data['income_limits'].columns:
            return jsonify({'error': 'Invalid income type or family size'}), 400
        
        # Create a simplified version with just fips and the income limit
        result = data['income_limits'][['fips', column_name]].copy()
        result = result.rename(columns={column_name: 'income_limit'})
        
        # Convert to records and handle NaN values
        records = result.to_dict('records')
        chatbot = AppalachianChatbot(data)
        records_converted = chatbot._convert_numpy_types(records)
        
        return jsonify(records_converted)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/table', methods=['POST'])
def export_table():
    """Export data table as Excel or CSV"""
    try:
        data_to_export = request.get_json().get('data', [])
        export_format = request.get_json().get('format', 'csv')  # 'csv' or 'excel'
        
        if not data_to_export:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(data_to_export)
        
        # Create BytesIO buffer
        buffer = BytesIO()
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if export_format == 'excel':
            # Export to Excel
            df.to_excel(buffer, index=False, engine='openpyxl')
            filename = f'appalachian_data_{timestamp}.xlsx'
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        else:
            # Export to CSV
            df.to_csv(buffer, index=False)
            filename = f'appalachian_data_{timestamp}.csv'
            mimetype = 'text/csv'
        
        buffer.seek(0)
        
        return send_file(
            buffer,
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Appalachian Map Chatbot with OpenAI integration...")
    print("Available data:")
    print(f"- Counties: {len(data['counties'])}")
    print(f"- Unemployment records: {len(data['unemployment'])}")
    print(f"- GDP records: {len(data['gdp'])}")
    print(f"- Summary records: {len(data['summary'])}")
    
    # Check OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key and api_key != 'your_openai_api_key_here':
        print("SUCCESS: OpenAI API key configured - Enhanced NLP enabled")
    else:
        print("WARNING: OpenAI API key not configured - Using fallback regex parsing")
        print("   Add your API key to .env file for better natural language processing")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
