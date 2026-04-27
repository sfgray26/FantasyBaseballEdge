"""
Weather integration for fantasy baseball decisions.

Weather is CRITICAL in baseball:
- Temperature: Every 10°F ≈ 3-4% more distance on fly balls
- Wind: 10mph out = 15-20% more HRs, 10mph in = 15-20% fewer
- Humidity: Actually helps ball carry (counterintuitive)
- Precipitation: Delays, postponements, pitcher changes
- Altitude: Coors Field effect
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

import requests

from backend.fantasy_baseball.elite_context import WeatherContext
from backend.services.cache_service import get_cache_service

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "weather"
CACHE_TTL_MINUTES = 30  # Weather changes quickly

# MLB venue to city mapping (for weather API)
VENUE_CITIES = {
    # AL East
    "Camden Yards": ("Baltimore", "MD"),
    "Fenway Park": ("Boston", "MA"),
    "Yankee Stadium": ("Bronx", "NY"),
    "Tropicana Field": ("St. Petersburg", "FL"),  # Dome
    "Rogers Centre": ("Toronto", "ON"),  # Retractable
    
    # AL Central
    "Guaranteed Rate Field": ("Chicago", "IL"),
    "Progressive Field": ("Cleveland", "OH"),
    "Comerica Park": ("Detroit", "MI"),
    "Kauffman Stadium": ("Kansas City", "MO"),
    "Target Field": ("Minneapolis", "MN"),
    
    # AL West
    "Minute Maid Park": ("Houston", "TX"),  # Retractable
    "Angel Stadium": ("Anaheim", "CA"),
    "Oakland Coliseum": ("Oakland", "CA"),
    "T-Mobile Park": ("Seattle", "WA"),  # Retractable
    "Globe Life Field": ("Arlington", "TX"),  # Dome
    
    # NL East
    "Truist Park": ("Atlanta", "GA"),
    "LoanDepot Park": ("Miami", "FL"),  # Dome
    "Citi Field": ("Flushing", "NY"),
    "Citizens Bank Park": ("Philadelphia", "PA"),
    "Nationals Park": ("Washington", "DC"),
    
    # NL Central
    "Wrigley Field": ("Chicago", "IL"),
    "Great American Ball Park": ("Cincinnati", "OH"),
    "American Family Field": ("Milwaukee", "WI"),  # Retractable
    "PNC Park": ("Pittsburgh", "PA"),
    "Busch Stadium": ("St. Louis", "MO"),
    
    # NL West
    "Chase Field": ("Phoenix", "AZ"),  # Retractable
    "Coors Field": ("Denver", "CO"),  # Altitude
    "Dodger Stadium": ("Los Angeles", "CA"),
    "Petco Park": ("San Diego", "CA"),
    "Oracle Park": ("San Francisco", "CA"),
}

# Team abbreviation to venue mapping (fallback when venue not provided)
TEAM_VENUES = {
    "BAL": "Camden Yards",
    "BOS": "Fenway Park",
    "NYY": "Yankee Stadium",
    "TB": "Tropicana Field",
    "TOR": "Rogers Centre",
    "CWS": "Guaranteed Rate Field",
    "CLE": "Progressive Field",
    "DET": "Comerica Park",
    "KC": "Kauffman Stadium",
    "MIN": "Target Field",
    "HOU": "Minute Maid Park",
    "LAA": "Angel Stadium",
    "OAK": "Oakland Coliseum",
    "SEA": "T-Mobile Park",
    "TEX": "Globe Life Field",
    "ATL": "Truist Park",
    "MIA": "LoanDepot Park",
    "NYM": "Citi Field",
    "PHI": "Citizens Bank Park",
    "WSH": "Nationals Park",
    "CHC": "Wrigley Field",
    "CIN": "Great American Ball Park",
    "MIL": "American Family Field",
    "PIT": "PNC Park",
    "STL": "Busch Stadium",
    "ARI": "Chase Field",
    "COL": "Coors Field",
    "LAD": "Dodger Stadium",
    "SD": "Petco Park",
    "SF": "Oracle Park",
}

# Reverse mapping: venue name to team abbreviation (for cache keys)
VENUE_TO_TEAM = {venue: team for team, venue in TEAM_VENUES.items()}

# Stadium characteristics
STADIUM_PROFILES = {
    "Coors Field": {"elevation": 5183, "park_factor": 1.35, "dome": False},
    "Fenway Park": {"elevation": 16, "park_factor": 1.15, "dome": False, "green_monster": True},
    "Yankee Stadium": {"elevation": 39, "park_factor": 1.10, "dome": False, "short_porch": True},
    "Oracle Park": {"elevation": 7, "park_factor": 0.85, "dome": False},
    "Petco Park": {"elevation": 23, "park_factor": 0.88, "dome": False},
    "Tropicana Field": {"elevation": 40, "park_factor": 0.95, "dome": True},
    "Rogers Centre": {"elevation": 268, "park_factor": 1.02, "retractable": True},
    "Minute Maid Park": {"elevation": 55, "park_factor": 1.05, "retractable": True},
    "T-Mobile Park": {"elevation": 17, "park_factor": 0.95, "retractable": True},
    "Chase Field": {"elevation": 1082, "park_factor": 1.05, "retractable": True},
    "American Family Field": {"elevation": 583, "park_factor": 1.00, "retractable": True},
    "Globe Life Field": {"elevation": 550, "park_factor": 0.98, "dome": True},
    "LoanDepot Park": {"elevation": 6, "park_factor": 0.92, "dome": True},
}


@dataclass
class GameWeather:
    """Complete weather for a specific game."""
    venue: str
    game_time: datetime
    
    # Temperature
    temperature: int = 72  # Fahrenheit
    feels_like: int = 72
    
    # Wind (CRITICAL for baseball)
    wind_speed: int = 0  # MPH
    wind_direction: str = ""  # "out", "in", "left", "right", "unknown"
    wind_gust: int = 0
    
    # Conditions
    condition: str = "Clear"  # "Clear", "Cloudy", "Rain", etc.
    precipitation_chance: int = 0  # 0-100%
    humidity: int = 50  # 0-100%
    
    # Stadium
    is_dome: bool = False
    roof_closed: Optional[bool] = None  # For retractable
    elevation: int = 0  # Feet
    
    # Derived scores
    hitter_friendly_score: float = 5.0  # 0-10
    hr_factor: float = 1.0  # Multiplier
    game_risk: str = "low"  # "low", "medium", "high", "postponement_risk"
    fallback_mode: bool = False
    
    def to_context(self) -> WeatherContext:
        """Convert to WeatherContext for elite context."""
        return WeatherContext(
            temperature=self.temperature,
            wind_speed=self.wind_speed,
            wind_direction=self.wind_direction,
            precipitation="rain" if self.precipitation_chance > 50 else "none",
            roof_closed=self.roof_closed or self.is_dome,
        )
    
    @property
    def summary(self) -> str:
        """One-line weather summary."""
        if self.is_dome:
            return f"🌐 Dome ({self.venue})"
        
        wind_emoji = "💨" if self.wind_speed > 15 else "🌬️" if self.wind_speed > 10 else ""
        wind_desc = f"{wind_emoji} {self.wind_speed}mph {self.wind_direction}" if self.wind_speed > 5 else "calm"
        
        rain_emoji = "🌧️" if self.precipitation_chance > 50 else "⛈️" if self.precipitation_chance > 30 else ""
        
        return f"{self.temperature}°F, {wind_desc} {rain_emoji}".strip()


class WeatherFetcher:
    """Fetch weather for MLB games."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._api_key_failed = False  # Circuit breaker: skip API calls after auth failure
        self._cache_service = get_cache_service()  # Redis caching layer
        
        # Try to get API key from environment
        if not self.api_key:
            import os
            self.api_key = os.getenv("OPENWEATHER_API_KEY")
        
        # Log warning if no API key
        if not self.api_key:
            logger.warning(
                "OPENWEATHER_API_KEY not set. Weather data will use seasonal estimates. "
                "Set OPENWEATHER_API_KEY in Railway environment variables."
            )
        else:
            logger.debug("WeatherFetcher initialized with API key")
    
    def get_game_weather(
        self, 
        venue: str, 
        game_time: datetime,
        team_abbr: Optional[str] = None
    ) -> GameWeather:
        """
        Get weather for a specific game.
        
        Args:
            venue: Stadium name (e.g., "Coors Field")
            game_time: When the game starts
            team_abbr: Team abbreviation (for stadium lookup fallback)
        """
        # Handle empty venue with team fallback
        if not venue or venue.strip() == "":
            if team_abbr and team_abbr.upper() in TEAM_VENUES:
                venue = TEAM_VENUES[team_abbr.upper()]
                logger.debug(f"Using venue '{venue}' from team {team_abbr}")
            else:
                logger.warning("Empty venue and no team provided, using neutral weather")
                return GameWeather(
                    venue="Unknown",
                    game_time=game_time,
                    temperature=72,
                )
        
        cache_key = f"{venue}_{game_time.strftime('%Y%m%d_%H%M')}"

        # Check Redis cache first (faster than filesystem)
        team_abbr_upper = team_abbr.upper() if team_abbr else None
        if team_abbr_upper and team_abbr_upper in TEAM_VENUES:
            game_date = game_time.strftime('%Y-%m-%d')
            redis_cached = self._cache_service.get_weather(team_abbr_upper, game_date)
            if redis_cached:
                logger.debug(f"Redis cache hit for {venue} on {game_date}")
                # Convert cached dict back to GameWeather object
                return GameWeather(**redis_cached)

        # Fall back to filesystem cache
        cached = self._load_cache(cache_key)
        if cached:
            return cached
        
        # Check if dome
        stadium_profile = STADIUM_PROFILES.get(venue, {})
        if stadium_profile.get("dome"):
            weather = GameWeather(
                venue=venue,
                game_time=game_time,
                is_dome=True,
                elevation=stadium_profile.get("elevation", 0),
                hitter_friendly_score=5.0,
                hr_factor=stadium_profile.get("park_factor", 1.0),
            )
            self._save_cache(cache_key, weather)
            # Also save to Redis if we have a team abbreviation
            if team_abbr and team_abbr.upper() in TEAM_VENUES:
                game_date = game_time.strftime('%Y-%m-%d')
                weather_dict = {
                    'venue': weather.venue,
                    'game_time': weather.game_time.isoformat(),
                    'temperature': weather.temperature,
                    'is_dome': weather.is_dome,
                    'elevation': weather.elevation,
                    'hitter_friendly_score': weather.hitter_friendly_score,
                    'hr_factor': weather.hr_factor,
                }
                self._cache_service.set_weather(team_abbr.upper(), game_date, weather_dict)
            return weather

        # Fetch real weather
        weather = self._fetch_weather(venue, game_time, stadium_profile)
        self._save_cache(cache_key, weather)
        # Also save to Redis if we have a team abbreviation
        if team_abbr and team_abbr.upper() in TEAM_VENUES:
            game_date = game_time.strftime('%Y-%m-%d')
            weather_dict = {
                'venue': weather.venue,
                'game_time': weather.game_time.isoformat(),
                'temperature': weather.temperature,
                'is_dome': weather.is_dome,
                'elevation': weather.elevation,
                'hitter_friendly_score': weather.hitter_friendly_score,
                'hr_factor': weather.hr_factor,
            }
            self._cache_service.set_weather(team_abbr.upper(), game_date, weather_dict)
        return weather
    
    def get_weather_for_games(
        self, 
        games: List[Dict]
    ) -> Dict[str, GameWeather]:
        """
        Fetch weather for multiple games efficiently.
        
        Args:
            games: List of dicts with 'venue', 'game_time', 'home_team'
        
        Returns:
            Dict mapping venue -> GameWeather
        """
        results = {}
        for game in games:
            venue = game.get("venue", "")
            game_time = game.get("game_time", datetime.now())
            team_abbr = game.get("home_team") or game.get("team")
            
            try:
                weather = self.get_game_weather(venue, game_time, team_abbr=team_abbr)
                results[venue] = weather
            except Exception as e:
                logger.warning(f"Failed to fetch weather for {venue}: {e}")
                # Return neutral weather on failure
                results[venue] = GameWeather(
                    venue=venue,
                    game_time=game_time,
                )
        
        return results
    
    def _fetch_weather(
        self, 
        venue: str, 
        game_time: datetime,
        stadium_profile: Dict
    ) -> GameWeather:
        """Fetch weather from API."""
        
        # Get location
        city_state = VENUE_CITIES.get(venue)
        if not city_state:
            logger.warning(f"Unknown venue: {venue}")
            return GameWeather(
                venue=venue,
                game_time=game_time,
                elevation=stadium_profile.get("elevation", 0),
            )
        
        city, state = city_state
        
        # Use OpenWeatherMap if API key available and not circuit-broken
        if self.api_key and not self._api_key_failed:
            return self._fetch_openweather(venue, game_time, city, state, stadium_profile)
        else:
            # Fallback: estimate based on stadium profile and season
            return self._estimate_weather(venue, game_time, stadium_profile)
    
    def _fetch_openweather(
        self,
        venue: str,
        game_time: datetime,
        city: str,
        state: str,
        stadium_profile: Dict
    ) -> GameWeather:
        """Fetch from OpenWeatherMap API.

        Strategy:
        1. Geocode city → lat/lon
        2. Try free-tier ``2.5/weather`` (current conditions) first — works with any key.
        3. Only attempt ``2.5/onecall`` if current-weather succeeded (proves the key is valid).
        4. On any 401/403, trip the circuit breaker so we don't hammer a dead key.
        """
        lat: float | None = None
        lon: float | None = None
        try:
            # Geocode
            geo_url = "http://api.openweathermap.org/geo/1.0/direct"
            geo_resp = self._session.get(
                geo_url,
                params={"q": f"{city},{state},US", "limit": 1, "appid": self.api_key},
                timeout=10
            )
            if geo_resp.status_code in (401, 403):
                self._api_key_failed = True
                logger.error("OpenWeather API key rejected during geocode — disabling live weather.")
                return self._estimate_weather(venue, game_time, stadium_profile)
            geo_resp.raise_for_status()
            geo_data = geo_resp.json()
            
            if not geo_data:
                raise ValueError(f"Could not geocode {city}, {state}")
            
            lat, lon = geo_data[0]["lat"], geo_data[0]["lon"]

            # --- Free-tier: current weather (always works with valid key) ---
            current_weather = self._fetch_openweather_current(
                venue=venue,
                game_time=game_time,
                lat=lat,
                lon=lon,
                stadium_profile=stadium_profile,
            )
            # If game_time is within ~2h of now, current conditions are good enough
            return current_weather
            
        except Exception as e:
            # Trip circuit breaker on auth failures so we don't hammer a dead key
            err_str = str(e)
            if "401" in err_str or "403" in err_str or "Unauthorized" in err_str:
                self._api_key_failed = True
                logger.error(
                    "OpenWeather API key rejected (401/403) — disabling live weather for this "
                    "session. Renew OPENWEATHER_API_KEY in Railway environment variables."
                )
            else:
                logger.warning(f"OpenWeather fetch failed for {venue}: {e}")
            return self._estimate_weather(venue, game_time, stadium_profile)

    def _fetch_openweather_current(
        self,
        venue: str,
        game_time: datetime,
        lat: float,
        lon: float,
        stadium_profile: Dict,
    ) -> GameWeather:
        """Fallback for free-tier keys: current conditions only."""
        weather_url = "https://api.openweathermap.org/data/2.5/weather"
        weather_resp = self._session.get(
            weather_url,
            params={
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "imperial",
            },
            timeout=10,
        )
        weather_resp.raise_for_status()
        data = weather_resp.json()

        main = data.get("main", {})
        wind = data.get("wind", {})
        weather_arr = data.get("weather", [{}])
        weather_main = weather_arr[0].get("main", "Clear") if weather_arr else "Clear"

        wind_speed = int(wind.get("speed", 0) or 0)
        wind_dir = self._degrees_to_direction(int(wind.get("deg", 0) or 0))
        wind_impact = self._calculate_wind_impact(venue, wind_dir, wind_speed)

        temp = float(main.get("temp", 72) or 72)
        feels_like = float(main.get("feels_like", temp) or temp)
        park_factor = stadium_profile.get("park_factor", 1.0)
        elevation = stadium_profile.get("elevation", 0)

        return GameWeather(
            venue=venue,
            game_time=game_time,
            temperature=int(temp),
            feels_like=int(feels_like),
            wind_speed=wind_speed,
            wind_direction=wind_dir,
            wind_gust=int(wind.get("gust", 0) or 0),
            condition=weather_main,
            precipitation_chance=0,
            humidity=int(main.get("humidity", 50) or 50),
            is_dome=False,
            elevation=elevation,
            hitter_friendly_score=self._calculate_hitter_score(
                temp,
                wind_speed,
                wind_impact,
                park_factor,
            ),
            hr_factor=self._calculate_hr_factor(
                temp,
                wind_speed,
                wind_impact,
                park_factor,
                elevation,
            ),
            fallback_mode=True,
        )
    
    def _estimate_weather(
        self,
        venue: str,
        game_time: datetime,
        stadium_profile: Dict
    ) -> GameWeather:
        """Estimate weather based on time of year and location."""
        month = game_time.month
        
        # Seasonal temperature estimates by region
        temp_estimates = {
            # Cold weather cities
            "Minneapolis": {4: 55, 5: 65, 6: 75, 7: 80, 8: 78, 9: 70, 10: 55},
            "Boston": {4: 58, 5: 68, 6: 77, 7: 82, 8: 80, 9: 72, 10: 60},
            "Chicago": {4: 58, 5: 70, 6: 80, 7: 84, 8: 82, 9: 74, 10: 60},
            "New York": {4: 60, 5: 70, 6: 78, 7: 83, 8: 81, 9: 74, 10: 62},
            "Toronto": {4: 52, 5: 63, 6: 72, 7: 77, 8: 75, 9: 68, 10: 55},
            
            # Warm weather cities
            "Atlanta": {4: 70, 5: 78, 6: 85, 7: 89, 8: 88, 9: 82, 10: 72},
            "Houston": {4: 75, 5: 82, 6: 89, 7: 94, 8: 94, 9: 88, 10: 78},
            "Phoenix": {4: 80, 5: 90, 6: 100, 7: 105, 8: 103, 9: 96, 10: 82},
            "Miami": {4: 78, 5: 82, 6: 86, 7: 88, 8: 88, 9: 86, 10: 80},
            "Los Angeles": {4: 68, 5: 72, 6: 78, 7: 84, 8: 85, 9: 82, 10: 75},
            "San Diego": {4: 65, 5: 68, 6: 72, 7: 76, 8: 78, 9: 76, 10: 70},
            "San Francisco": {4: 60, 5: 64, 6: 68, 7: 70, 8: 70, 9: 72, 10: 65},
        }
        
        # Get city from venue
        city_state = VENUE_CITIES.get(venue, ("", ""))
        city = city_state[0]
        
        # Default to 72°F if unknown
        temp = 72
        for city_name, temps in temp_estimates.items():
            if city_name in city:
                temp = temps.get(month, 72)
                break
        
        elevation = stadium_profile.get("elevation", 0)
        park_factor = stadium_profile.get("park_factor", 1.0)
        wind_speed = 0
        wind_impact = "neutral"
        
        return GameWeather(
            venue=venue,
            game_time=game_time,
            temperature=temp,
            elevation=elevation,
            wind_speed=wind_speed,
            wind_direction=wind_impact,
            hitter_friendly_score=self._calculate_hitter_score(
                temp,
                wind_speed,
                wind_impact,
                park_factor,
            ),
            hr_factor=self._calculate_hr_factor(
                temp,
                wind_speed,
                wind_impact,
                park_factor,
                elevation,
            ),
            fallback_mode=True,
        )
    
    def _degrees_to_direction(self, degrees: int) -> str:
        """Convert wind degrees to cardinal direction."""
        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                     "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        index = round(degrees / 22.5) % 16
        return directions[index]
    
    def _calculate_wind_impact(self, venue: str, wind_dir: str, wind_speed: int) -> str:
        """
        Determine if wind helps or hurts hitters.
        
        This is stadium-specific based on orientation.
        Simplified: assume most stadiums face roughly east.
        """
        if wind_speed < 5:
            return "neutral"
        
        # Simplified: out = helping, in = hurting
        # In reality, this depends on stadium orientation
        if wind_dir in ["S", "SSW", "SW"]:
            return "out"  # Helping
        elif wind_dir in ["N", "NNE", "NNW"]:
            return "in"  # Hurting
        else:
            return "cross"  # Sideways, less impact
    
    def _calculate_hitter_score(
        self,
        temp: float,
        wind_speed: float,
        wind_impact: str,
        park_factor: float
    ) -> float:
        """Calculate 0-10 hitter-friendly score."""
        score = 5.0
        
        # Temperature (warmer = better)
        score += (temp - 72) / 10
        
        # Wind
        if wind_impact == "out":
            score += wind_speed / 5
        elif wind_impact == "in":
            score -= wind_speed / 5
        
        # Park factor
        score += (park_factor - 1.0) * 5
        
        return max(0, min(10, score))
    
    def _calculate_hr_factor(
        self,
        temp: float,
        wind_speed: float,
        wind_impact: str,
        park_factor: float,
        elevation: int
    ) -> float:
        """Calculate HR factor (1.0 = neutral, 1.5 = 50% more HRs)."""
        factor = 1.0
        
        # Temperature: every 10°F ≈ 3-4% more distance
        factor += (temp - 72) / 10 * 0.035
        
        # Wind: 10mph out ≈ 15% more HRs
        if wind_impact == "out":
            factor += wind_speed / 10 * 0.15
        elif wind_impact == "in":
            factor -= wind_speed / 10 * 0.15
        
        # Park factor
        factor *= park_factor
        
        # Elevation (Coors Field effect)
        if elevation > 3000:
            factor *= 1.15  # ~15% boost at altitude
        
        return round(factor, 2)
    
    def _assess_game_risk(
        self,
        weather_data: Dict,
        stadium_profile: Dict
    ) -> str:
        """Assess risk of delay/postponement."""
        if stadium_profile.get("dome") or stadium_profile.get("roof_closed"):
            return "low"
        
        precip_chance = weather_data.get("pop", 0) * 100
        condition = weather_data.get("weather", [{}])[0].get("main", "").lower()
        
        if precip_chance > 70 or "thunderstorm" in condition:
            return "postponement_risk"
        elif precip_chance > 50 or "rain" in condition:
            return "high"
        elif precip_chance > 30:
            return "medium"
        else:
            return "low"
    
    def _load_cache(self, cache_key: str) -> Optional[GameWeather]:
        """Load from cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None
        
        try:
            data = json.loads(cache_file.read_text())
            cached_time = datetime.fromisoformat(data["cached_at"])
            
            if datetime.now() - cached_time > timedelta(minutes=CACHE_TTL_MINUTES):
                return None
            
            return GameWeather(**data["data"])
        except Exception as e:
            logger.debug(f"Cache load failed: {e}")
            return None
    
    def _save_cache(self, cache_key: str, weather: GameWeather) -> None:
        """Save to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            data = {
                "cached_at": datetime.now().isoformat(),
                "data": {
                    "venue": weather.venue,
                    "game_time": weather.game_time.isoformat(),
                    "temperature": weather.temperature,
                    "feels_like": weather.feels_like,
                    "wind_speed": weather.wind_speed,
                    "wind_direction": weather.wind_direction,
                    "wind_gust": weather.wind_gust,
                    "condition": weather.condition,
                    "precipitation_chance": weather.precipitation_chance,
                    "humidity": weather.humidity,
                    "is_dome": weather.is_dome,
                    "roof_closed": weather.roof_closed,
                    "elevation": weather.elevation,
                    "hitter_friendly_score": weather.hitter_friendly_score,
                    "hr_factor": weather.hr_factor,
                    "game_risk": weather.game_risk,
                }
            }
            cache_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"Cache save failed: {e}")


def get_weather_fetcher(api_key: Optional[str] = None) -> WeatherFetcher:
    """Factory function."""
    return WeatherFetcher(api_key)


def validate_weather_api_key(api_key: Optional[str] = None) -> Tuple[bool, str]:
    """
    Validate OpenWeather API key.
    
    Returns:
        (is_valid, message)
    """
    if not api_key:
        import os
        api_key = os.getenv("OPENWEATHER_API_KEY")
    
    if not api_key:
        return False, "OPENWEATHER_API_KEY not set in environment"
    
    # Test API call
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": "New York,NY,US",
            "appid": api_key,
        }
        resp = requests.get(url, params=params, timeout=10)
        
        if resp.status_code == 401:
            return False, "Invalid API key (401 Unauthorized)"
        elif resp.status_code != 200:
            return False, f"API error: {resp.status_code}"
        
        return True, "API key valid"
        
    except Exception as e:
        return False, f"Failed to validate API key: {e}"
