"""
Park-specific weather analysis for MLB stadiums.

Each stadium has unique orientation and microclimate.
Wind direction matters differently at each park based on:
- Which way the stadium faces (home plate to center field)
- Local wind patterns
- Surrounding geography
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

from backend.fantasy_baseball.weather_fetcher import GameWeather

logger = logging.getLogger(__name__)


class WindImpact(Enum):
    """How wind affects a stadium."""
    HELPS_HITTERS = "helps"
    HURTS_HITTERS = "hurts"
    NEUTRAL = "neutral"
    CROSS_WIND = "cross"
    UNPREDICTABLE = "unpredictable"


@dataclass
class StadiumOrientation:
    """Physical orientation of a stadium."""
    # Degrees from north (0-360)
    # 0 = due north, 90 = east, 180 = south, 270 = west
    home_to_center: int
    
    # Wind patterns that help/hurt
    # Based on prevailing winds and stadium geometry
    helps_winds: List[str] = None  # e.g., ["SW", "S"]
    hurts_winds: List[str] = None  # e.g., ["NE", "N"]
    
    def __post_init__(self):
        if self.helps_winds is None:
            self.helps_winds = []
        if self.hurts_winds is None:
            self.hurts_winds = []


@dataclass
class ParkWeatherProfile:
    """Complete weather profile for a stadium."""
    venue: str
    orientation: StadiumOrientation
    
    # Microclimate factors
    elevation: int = 0  # Feet
    humidity_factor: float = 1.0  # Coastal vs inland
    temperature_bias: int = 0  # Degrees vs regional average
    
    # Wind effects (multipliers when wind helps/hurts)
    wind_help_boost: float = 0.15  # 15% boost when wind helps
    wind_hurt_penalty: float = -0.15  # 15% penalty when wind hurts
    
    # Special characteristics
    short_porch_left: bool = False
    short_porch_right: bool = False
    asymmetrical: bool = False  # e.g., Fenway, Yankee Stadium
    wind_swallower: bool = False  # Stadium blocks wind
    wind_tunnel: bool = False  # Wind gets concentrated
    
    def calculate_wind_impact(self, wind_dir: str, wind_speed: int) -> Tuple[WindImpact, float]:
        """
        Calculate how wind affects this specific park.
        
        Returns:
            (impact_type, multiplier)
        """
        if wind_speed < 5:
            return WindImpact.NEUTRAL, 1.0
        
        # Check if wind helps
        if any(help_dir in wind_dir for help_dir in self.orientation.helps_winds):
            return WindImpact.HELPS_HITTERS, 1.0 + (self.wind_help_boost * wind_speed / 10)
        
        # Check if wind hurts
        if any(hurt_dir in wind_dir for hurt_dir in self.orientation.hurts_winds):
            return WindImpact.HURTS_HITTERS, 1.0 + (self.wind_hurt_penalty * wind_speed / 10)
        
        # Cross wind or unpredictable
        if self.wind_swallower:
            return WindImpact.NEUTRAL, 1.0  # Stadium blocks it
        if self.wind_tunnel:
            return WindImpact.UNPREDICTABLE, 1.0  # Turbulence
        
        return WindImpact.CROSS_WIND, 1.0


# Stadium orientations (home plate to center field)
# and prevailing wind effects
STADIUM_PROFILES = {
    # AL East
    "Camden Yards": ParkWeatherProfile(
        venue="Camden Yards",
        orientation=StadiumOrientation(
            home_to_center=10,  # Slightly east of due north
            helps_winds=["S", "SW", "WSW"],
            hurts_winds=["N", "NE", "ENE"]
        ),
        elevation=40,
        short_porch_right=True,  # 318 ft down the line
    ),
    
    "Fenway Park": ParkWeatherProfile(
        venue="Fenway Park",
        orientation=StadiumOrientation(
            home_to_center=115,  # Southeast - Green Monster in left
            helps_winds=["NW", "WNW", "W"],
            hurts_winds=["SE", "SSE", "E"]
        ),
        elevation=16,
        asymmetrical=True,
        short_porch_left=True,  # 310 ft to Green Monster
        wind_swallower=True,  # Green Monster blocks some wind
    ),
    
    "Yankee Stadium": ParkWeatherProfile(
        venue="Yankee Stadium",
        orientation=StadiumOrientation(
            home_to_center=15,  # Just east of north
            helps_winds=["S", "SSW", "SW"],
            hurts_winds=["N", "NNE", "NE"]
        ),
        elevation=39,
        short_porch_right=True,  # 314 ft
        wind_tunnel=True,  # Near Harlem River
    ),
    
    "Tropicana Field": ParkWeatherProfile(
        venue="Tropicana Field",
        orientation=StadiumOrientation(home_to_center=0),
        elevation=40,
        # Dome - weather doesn't matter
    ),
    
    "Rogers Centre": ParkWeatherProfile(
        venue="Rogers Centre",
        orientation=StadiumOrientation(
            home_to_center=110,  # Southeast
            helps_winds=["W", "WNW"],
            hurts_winds=["E", "ESE"]
        ),
        elevation=268,
        # Retractable roof
    ),
    
    # AL Central
    "Guaranteed Rate Field": ParkWeatherProfile(
        venue="Guaranteed Rate Field",
        orientation=StadiumOrientation(
            home_to_center=125,  # Southeast
            helps_winds=["NW", "W"],
            hurts_winds=["SE", "E"]
        ),
        elevation=595,
        wind_tunnel=True,  # Wind off Lake Michigan
    ),
    
    "Progressive Field": ParkWeatherProfile(
        venue="Progressive Field",
        orientation=StadiumOrientation(
            home_to_center=60,  # Northeast
            helps_winds=["SW", "WSW"],
            hurts_winds=["NE", "NNE"]
        ),
        elevation=673,
    ),
    
    "Comerica Park": ParkWeatherProfile(
        venue="Comerica Park",
        orientation=StadiumOrientation(
            home_to_center=105,  # East-southeast
            helps_winds=["W", "WNW"],
            hurts_winds=["E", "ESE"]
        ),
        elevation=602,
    ),
    
    "Kauffman Stadium": ParkWeatherProfile(
        venue="Kauffman Stadium",
        orientation=StadiumOrientation(
            home_to_center=5,  # Due north
            helps_winds=["S", "SSW"],
            hurts_winds=["N", "NNE"]
        ),
        elevation=886,
    ),
    
    "Target Field": ParkWeatherProfile(
        venue="Target Field",
        orientation=StadiumOrientation(
            home_to_center=85,  # East
            helps_winds=["W", "WSW"],
            hurts_winds=["E", "ENE"]
        ),
        elevation=840,
    ),
    
    # AL West
    "Minute Maid Park": ParkWeatherProfile(
        venue="Minute Maid Park",
        orientation=StadiumOrientation(
            home_to_center=15,  # North-northeast
            helps_winds=["S", "SW"],
            hurts_winds=["N", "NE"]
        ),
        elevation=55,
        short_porch_left=True,  # Crawford Boxes
        # Retractable roof
    ),
    
    "Angel Stadium": ParkWeatherProfile(
        venue="Angel Stadium",
        orientation=StadiumOrientation(
            home_to_center=340,  # North-northwest
            helps_winds=["S", "SE"],
            hurts_winds=["N", "NW"]
        ),
        elevation=154,
    ),
    
    "Oakland Coliseum": ParkWeatherProfile(
        venue="Oakland Coliseum",
        orientation=StadiumOrientation(
            home_to_center=40,  # Northeast
            helps_winds=["SW", "WSW"],
            hurts_winds=["NE", "NNE"]
        ),
        elevation=0,  # Sea level
        humidity_factor=1.1,  # Coastal
        wind_tunnel=True,  # Often swirls
    ),
    
    "T-Mobile Park": ParkWeatherProfile(
        venue="T-Mobile Park",
        orientation=StadiumOrientation(
            home_to_center=90,  # Due east
            helps_winds=["W", "WSW"],
            hurts_winds=["E", "ESE"]
        ),
        elevation=17,
        humidity_factor=1.15,  # Coastal marine layer
        # Retractable roof
    ),
    
    "Globe Life Field": ParkWeatherProfile(
        venue="Globe Life Field",
        orientation=StadiumOrientation(home_to_center=0),
        elevation=550,
        # Dome
    ),
    
    # NL East
    "Truist Park": ParkWeatherProfile(
        venue="Truist Park",
        orientation=StadiumOrientation(
            home_to_center=155,  # South-southeast
            helps_winds=["N", "NW"],
            hurts_winds=["S", "SSE"]
        ),
        elevation=981,
    ),
    
    "LoanDepot Park": ParkWeatherProfile(
        venue="LoanDepot Park",
        orientation=StadiumOrientation(home_to_center=0),
        elevation=6,
        # Dome
    ),
    
    "Citi Field": ParkWeatherProfile(
        venue="Citi Field",
        orientation=StadiumOrientation(
            home_to_center=15,  # North-northeast
            helps_winds=["S", "SW"],
            hurts_winds=["N", "NE"]
        ),
        elevation=13,
    ),
    
    "Citizens Bank Park": ParkWeatherProfile(
        venue="Citizens Bank Park",
        orientation=StadiumOrientation(
            home_to_center=15,  # North-northeast
            helps_winds=["S", "SSW"],
            hurts_winds=["N", "NNE"]
        ),
        elevation=19,
        short_porch_left=True,  # 330 ft
    ),
    
    "Nationals Park": ParkWeatherProfile(
        venue="Nationals Park",
        orientation=StadiumOrientation(
            home_to_center=65,  # Northeast
            helps_winds=["SW", "W"],
            hurts_winds=["NE", "E"]
        ),
        elevation=21,
    ),
    
    # NL Central
    "Wrigley Field": ParkWeatherProfile(
        venue="Wrigley Field",
        orientation=StadiumOrientation(
            home_to_center=40,  # Northeast
            helps_winds=["SW", "W", "WSW"],  # Famous wind out to right
            hurts_winds=["NE", "E", "ENE"]   # Wind in from right
        ),
        elevation=597,
        wind_tunnel=True,  # Lake effect + buildings
        asymmetrical=True,
    ),
    
    "Great American Ball Park": ParkWeatherProfile(
        venue="Great American Ball Park",
        orientation=StadiumOrientation(
            home_to_center=125,  # Southeast
            helps_winds=["NW", "W", "WNW"],
            hurts_winds=["SE", "E"]
        ),
        elevation=489,
        short_porch_right=True,  # 325 ft
    ),
    
    "American Family Field": ParkWeatherProfile(
        venue="American Family Field",
        orientation=StadiumOrientation(
            home_to_center=85,  # East
            helps_winds=["W", "WNW"],
            hurts_winds=["E", "ESE"]
        ),
        elevation=583,
        # Retractable roof
    ),
    
    "PNC Park": ParkWeatherProfile(
        venue="PNC Park",
        orientation=StadiumOrientation(
            home_to_center=65,  # Northeast
            helps_winds=["SW", "WSW"],
            hurts_winds=["NE", "E"]
        ),
        elevation=724,
    ),
    
    "Busch Stadium": ParkWeatherProfile(
        venue="Busch Stadium",
        orientation=StadiumOrientation(
            home_to_center=60,  # Northeast
            helps_winds=["SW", "W"],
            hurts_winds=["NE", "E"]
        ),
        elevation=435,
    ),
    
    # NL West
    "Chase Field": ParkWeatherProfile(
        venue="Chase Field",
        orientation=StadiumOrientation(
            home_to_center=105,  # East-southeast
            helps_winds=["W", "WNW"],
            hurts_winds=["E", "ESE"]
        ),
        elevation=1082,
        temperature_bias=+15,  # Hot
        # Retractable roof
    ),
    
    "Coors Field": ParkWeatherProfile(
        venue="Coors Field",
        orientation=StadiumOrientation(
            home_to_center=90,  # Due east
            helps_winds=["W", "WSW"],  # West = out to right field
            hurts_winds=["E", "ENE"]
        ),
        elevation=5183,  # Thin air
        temperature_bias=+10,  # Usually nice
        # Altitude is the story here
    ),
    
    "Dodger Stadium": ParkWeatherProfile(
        venue="Dodger Stadium",
        orientation=StadiumOrientation(
            home_to_center=25,  # North-northeast
            helps_winds=["S", "SSW"],
            hurts_winds=["N", "NNE"]
        ),
        elevation=501,
    ),
    
    "Petco Park": ParkWeatherProfile(
        venue="Petco Park",
        orientation=StadiumOrientation(
            home_to_center=115,  # Southeast
            helps_winds=["NW", "W", "WNW"],
            hurts_winds=["SE", "E"]
        ),
        elevation=23,
        humidity_factor=1.1,  # Coastal
        wind_tunnel=True,  # Near water
    ),
    
    "Oracle Park": ParkWeatherProfile(
        venue="Oracle Park",
        orientation=StadiumOrientation(
            home_to_center=115,  # Southeast - right field faces water
            helps_winds=["NW", "W"],  # Off the water = out
            hurts_winds=["SE", "E"]
        ),
        elevation=7,
        humidity_factor=1.15,  # Heavy marine layer
        wind_swallower=True,  # Right field wall blocks wind
        temperature_bias=-10,  # Cold marine layer
    ),
}


class ParkWeatherAnalyzer:
    """Analyze weather effects specific to each park."""
    
    def __init__(self):
        self.profiles = STADIUM_PROFILES
    
    def analyze_game(self, venue: str, weather: GameWeather) -> Dict:
        """
        Complete weather analysis for a specific game.
        
        Returns dict with:
            - wind_impact: HELPS/HURTS/NEUTRAL
            - wind_multiplier: HR multiplier from wind
            - temp_factor: Temperature effect
            - humidity_factor: Humidity effect
            - total_hr_factor: Combined multiplier
            - description: Human-readable summary
        """
        profile = self.profiles.get(venue)
        if not profile:
            logger.warning(f"No profile for {venue}, using generic")
            return self._generic_analysis(weather)
        
        # Wind analysis
        wind_impact, wind_mult = profile.calculate_wind_impact(
            weather.wind_direction, weather.wind_speed
        )
        
        # Temperature (warmer = better)
        temp_factor = 1.0 + (weather.temperature - 72) / 300
        
        # Humidity (higher = slightly better ball carry)
        # Counterintuitive but true - humid air is less dense
        humidity_factor = 1.0 + (weather.humidity - 50) / 1000
        humidity_factor *= profile.humidity_factor
        
        # Altitude (Coors Field effect)
        altitude_factor = 1.0 + (profile.elevation / 10000)
        
        # Park factor
        park_factor = 1.0 + (weather.park_factor - 1.0)
        
        # Combined
        total_hr_factor = wind_mult * temp_factor * humidity_factor * altitude_factor * park_factor
        
        # Build description
        desc_parts = []
        
        if venue == "Coors Field":
            desc_parts.append("⚠️ COORS FIELD - Extreme altitude boost")
        
        if wind_impact == WindImpact.HELPS_HITTERS:
            desc_parts.append(f"💨 Wind helps ({weather.wind_speed}mph {weather.wind_direction})")
        elif wind_impact == WindImpact.HURTS_HITTERS:
            desc_parts.append(f"🌬️ Wind hurts ({weather.wind_speed}mph {weather.wind_direction})")
        
        if weather.temperature > 85:
            desc_parts.append(f"☀️ Hot ({weather.temperature}°F)")
        elif weather.temperature < 50:
            desc_parts.append(f"❄️ Cold ({weather.temperature}°F)")
        
        if profile.elevation > 3000 and venue != "Coors Field":
            desc_parts.append(f"⛰️ High altitude ({profile.elevation:,}ft)")
        
        if weather.precipitation_chance > 30:
            desc_parts.append(f"🌧️ {weather.precipitation_chance}% rain")
        
        description = " | ".join(desc_parts) if desc_parts else "Neutral conditions"
        
        return {
            "venue": venue,
            "wind_impact": wind_impact.value,
            "wind_multiplier": round(wind_mult, 2),
            "temp_factor": round(temp_factor, 3),
            "humidity_factor": round(humidity_factor, 3),
            "altitude_factor": round(altitude_factor, 3),
            "park_factor": round(park_factor, 3),
            "total_hr_factor": round(total_hr_factor, 2),
            "description": description,
            "profile": profile,
        }
    
    def get_start_sit_recommendation(self, venue: str, weather: GameWeather) -> str:
        """Get start/sit recommendation based on weather."""
        analysis = self.analyze_game(venue, weather)
        hr_factor = analysis["total_hr_factor"]
        
        if hr_factor > 1.3:
            return "🚀 START power hitters - extreme hitting conditions"
        elif hr_factor > 1.15:
            return "📈 Favor power hitters - good hitting weather"
        elif hr_factor < 0.75:
            return "🛑 SIT power hitters - extreme suppression"
        elif hr_factor < 0.9:
            return "📉 Avoid power hitters - tough conditions"
        elif weather.game_risk in ["high", "postponement_risk"]:
            return f"⏳ MONITOR - {weather.game_risk.replace('_', ' ')}"
        else:
            return "⚖️ Neutral - play your normal lineup"
    
    def _generic_analysis(self, weather: GameWeather) -> Dict:
        """Fallback for unknown venues."""
        temp_factor = 1.0 + (weather.temperature - 72) / 300
        altitude_factor = 1.0 + (weather.elevation / 10000)
        
        return {
            "venue": weather.venue,
            "wind_impact": "unknown",
            "wind_multiplier": 1.0,
            "temp_factor": round(temp_factor, 3),
            "humidity_factor": 1.0,
            "altitude_factor": round(altitude_factor, 3),
            "park_factor": weather.park_factor,
            "total_hr_factor": round(temp_factor * altitude_factor * weather.park_factor, 2),
            "description": f"Generic analysis for {weather.venue}",
            "profile": None,
        }


def get_park_analyzer() -> ParkWeatherAnalyzer:
    """Factory function."""
    return ParkWeatherAnalyzer()
