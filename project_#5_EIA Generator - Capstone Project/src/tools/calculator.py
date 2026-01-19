"""
Environmental Calculators for EIA.

Calculates emissions, impacts, and other environmental metrics.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger


# =============================================================================
# Emission Factors
# =============================================================================

class FuelType(str, Enum):
    """Types of fuel."""
    DIESEL = "diesel"
    GASOLINE = "gasoline"
    COAL = "coal"
    NATURAL_GAS = "natural_gas"
    LPG = "lpg"
    ELECTRICITY = "electricity"


# Emission factors (kg CO2 per unit)
EMISSION_FACTORS = {
    FuelType.DIESEL: {"unit": "liter", "co2": 2.68, "nox": 0.04, "pm": 0.001},
    FuelType.GASOLINE: {"unit": "liter", "co2": 2.31, "nox": 0.02, "pm": 0.0005},
    FuelType.COAL: {"unit": "kg", "co2": 2.42, "nox": 0.02, "pm": 0.01},
    FuelType.NATURAL_GAS: {"unit": "m3", "co2": 1.89, "nox": 0.002, "pm": 0.0001},
    FuelType.LPG: {"unit": "kg", "co2": 2.98, "nox": 0.01, "pm": 0.0003},
    FuelType.ELECTRICITY: {"unit": "kWh", "co2": 0.5, "nox": 0.001, "pm": 0.0001},  # Vietnam grid average
}


@dataclass
class EmissionResult:
    """Emission calculation result."""
    co2_kg: float
    co2_tons: float
    nox_kg: float
    pm_kg: float
    fuel_type: str
    fuel_amount: float
    fuel_unit: str


# =============================================================================
# Emission Calculator
# =============================================================================

class EmissionCalculator:
    """
    Calculator for greenhouse gas and pollutant emissions.
    
    Based on IPCC emission factors and Vietnamese inventory guidelines.
    """
    
    def __init__(self):
        self.emission_factors = EMISSION_FACTORS
    
    def calculate_fuel_emissions(
        self,
        fuel_type: FuelType,
        amount: float,
    ) -> EmissionResult:
        """
        Calculate emissions from fuel consumption.
        
        Args:
            fuel_type: Type of fuel
            amount: Amount of fuel consumed
            
        Returns:
            Emission result
        """
        factors = self.emission_factors.get(fuel_type, {})
        
        co2_kg = amount * factors.get("co2", 0)
        nox_kg = amount * factors.get("nox", 0)
        pm_kg = amount * factors.get("pm", 0)
        
        return EmissionResult(
            co2_kg=co2_kg,
            co2_tons=co2_kg / 1000,
            nox_kg=nox_kg,
            pm_kg=pm_kg,
            fuel_type=fuel_type.value,
            fuel_amount=amount,
            fuel_unit=factors.get("unit", "unit"),
        )
    
    def calculate_construction_emissions(
        self,
        area_hectares: float,
        construction_months: int,
        equipment_hours_per_day: float = 8,
        diesel_liters_per_hour: float = 15,
    ) -> Dict[str, float]:
        """
        Estimate construction phase emissions.
        
        Args:
            area_hectares: Project area
            construction_months: Duration of construction
            equipment_hours_per_day: Hours equipment operates daily
            diesel_liters_per_hour: Average diesel consumption
            
        Returns:
            Dictionary with emission estimates
        """
        # Working days (22 days per month average)
        working_days = construction_months * 22
        
        # Equipment hours based on project size (more area = more equipment)
        equipment_count = max(1, area_hectares / 10)  # 1 equipment per 10 ha
        total_equipment_hours = working_days * equipment_hours_per_day * equipment_count
        
        # Diesel consumption
        total_diesel = total_equipment_hours * diesel_liters_per_hour
        
        # Calculate emissions
        result = self.calculate_fuel_emissions(FuelType.DIESEL, total_diesel)
        
        # Additional dust from earthwork (rough estimate: 0.5 kg/m2/month)
        dust_kg = area_hectares * 10000 * 0.5 * construction_months / 30
        
        return {
            "co2_tons": result.co2_tons,
            "nox_kg": result.nox_kg,
            "pm_kg": result.pm_kg,
            "dust_kg": dust_kg,
            "diesel_liters": total_diesel,
            "equipment_hours": total_equipment_hours,
        }
    
    def calculate_operation_emissions(
        self,
        electricity_kwh_year: float,
        fuel_consumption: Optional[Dict[str, float]] = None,
        vehicles: Optional[int] = None,
        vehicle_km_per_day: float = 50,
    ) -> Dict[str, float]:
        """
        Calculate annual operation phase emissions.
        
        Args:
            electricity_kwh_year: Annual electricity consumption
            fuel_consumption: Dictionary of fuel type -> annual consumption
            vehicles: Number of vehicles
            vehicle_km_per_day: Average km per vehicle per day
            
        Returns:
            Dictionary with emission estimates
        """
        emissions = {
            "co2_tons": 0,
            "nox_kg": 0,
            "pm_kg": 0,
        }
        
        # Electricity emissions
        elec_result = self.calculate_fuel_emissions(
            FuelType.ELECTRICITY, electricity_kwh_year
        )
        emissions["co2_tons"] += elec_result.co2_tons
        emissions["electricity_kwh"] = electricity_kwh_year
        
        # Fuel consumption
        if fuel_consumption:
            for fuel_str, amount in fuel_consumption.items():
                try:
                    fuel_type = FuelType(fuel_str)
                    result = self.calculate_fuel_emissions(fuel_type, amount)
                    emissions["co2_tons"] += result.co2_tons
                    emissions["nox_kg"] += result.nox_kg
                    emissions["pm_kg"] += result.pm_kg
                except ValueError:
                    logger.warning(f"Unknown fuel type: {fuel_str}")
        
        # Vehicle emissions
        if vehicles:
            # Assume diesel vehicles, 0.12 L/km
            annual_km = vehicles * vehicle_km_per_day * 250  # 250 working days
            diesel_liters = annual_km * 0.12
            vehicle_result = self.calculate_fuel_emissions(FuelType.DIESEL, diesel_liters)
            emissions["co2_tons"] += vehicle_result.co2_tons
            emissions["nox_kg"] += vehicle_result.nox_kg
            emissions["vehicle_km"] = annual_km
        
        return emissions
    
    def calculate_solar_benefits(
        self,
        capacity_mw: float,
        capacity_factor: float = 0.18,
        years: int = 25,
    ) -> Dict[str, float]:
        """
        Calculate emission reductions from solar power.
        
        Args:
            capacity_mw: Installed capacity in MW
            capacity_factor: Capacity factor (default 18% for Vietnam)
            years: Operating years
            
        Returns:
            Dictionary with avoided emissions
        """
        # Annual generation
        annual_mwh = capacity_mw * capacity_factor * 8760
        annual_kwh = annual_mwh * 1000
        
        # Grid emission factor for Vietnam (kg CO2/kWh)
        grid_factor = 0.7  # Approximate for Vietnam grid
        
        annual_co2_avoided = annual_kwh * grid_factor / 1000  # tons
        lifetime_co2_avoided = annual_co2_avoided * years
        
        return {
            "annual_generation_mwh": annual_mwh,
            "annual_co2_avoided_tons": annual_co2_avoided,
            "lifetime_co2_avoided_tons": lifetime_co2_avoided,
            "equivalent_trees": int(lifetime_co2_avoided * 45),  # ~45 trees per ton CO2
        }


# =============================================================================
# Impact Calculator
# =============================================================================

class ImpactCalculator:
    """
    Calculator for environmental impact assessment.
    
    Provides quantitative estimates for various impact categories.
    """
    
    def calculate_water_consumption(
        self,
        workers: int,
        process_water_m3_day: float = 0,
    ) -> Dict[str, float]:
        """
        Calculate water consumption.
        
        Args:
            workers: Number of workers
            process_water_m3_day: Industrial process water usage
            
        Returns:
            Dictionary with water consumption estimates
        """
        # Domestic water: 100 L/person/day
        domestic_m3_day = workers * 0.1
        
        total_m3_day = domestic_m3_day + process_water_m3_day
        annual_m3 = total_m3_day * 300  # 300 working days
        
        return {
            "domestic_m3_day": domestic_m3_day,
            "process_m3_day": process_water_m3_day,
            "total_m3_day": total_m3_day,
            "annual_m3": annual_m3,
        }
    
    def calculate_wastewater(
        self,
        water_consumption_m3_day: float,
        return_factor: float = 0.8,
    ) -> Dict[str, float]:
        """
        Calculate wastewater generation.
        
        Args:
            water_consumption_m3_day: Daily water consumption
            return_factor: Fraction of water becoming wastewater
            
        Returns:
            Dictionary with wastewater estimates
        """
        daily_m3 = water_consumption_m3_day * return_factor
        annual_m3 = daily_m3 * 300
        
        # Typical pollutant loads for mixed wastewater (mg/L)
        pollutant_loads = {
            "BOD5": 200,  # mg/L
            "COD": 400,
            "TSS": 250,
            "TN": 40,
            "TP": 8,
        }
        
        # Calculate daily pollutant load (kg/day)
        daily_loads = {}
        for pollutant, concentration in pollutant_loads.items():
            daily_loads[f"{pollutant}_kg_day"] = daily_m3 * concentration / 1000
        
        return {
            "daily_m3": daily_m3,
            "annual_m3": annual_m3,
            **daily_loads,
        }
    
    def calculate_solid_waste(
        self,
        workers: int,
        production_waste_tons_day: float = 0,
    ) -> Dict[str, float]:
        """
        Calculate solid waste generation.
        
        Args:
            workers: Number of workers
            production_waste_tons_day: Industrial production waste
            
        Returns:
            Dictionary with waste estimates
        """
        # Domestic waste: 0.5 kg/person/day
        domestic_kg_day = workers * 0.5
        domestic_tons_day = domestic_kg_day / 1000
        
        total_tons_day = domestic_tons_day + production_waste_tons_day
        annual_tons = total_tons_day * 300
        
        return {
            "domestic_kg_day": domestic_kg_day,
            "production_tons_day": production_waste_tons_day,
            "total_tons_day": total_tons_day,
            "annual_tons": annual_tons,
        }
    
    def calculate_noise_impact(
        self,
        source_db: float,
        distance_m: float,
        barriers: int = 0,
    ) -> Dict[str, float]:
        """
        Estimate noise level at distance.
        
        Args:
            source_db: Noise level at source (dBA)
            distance_m: Distance from source (m)
            barriers: Number of barriers/walls
            
        Returns:
            Dictionary with noise estimates
        """
        # Distance attenuation (spherical spreading)
        # Reduction = 20 * log10(d2/d1), reference d1 = 1m
        distance_reduction = 20 * (distance_m ** 0.5).bit_length()  # Simplified
        
        # Barrier attenuation (approximately 5 dBA per barrier)
        barrier_reduction = barriers * 5
        
        # Vegetation attenuation (assume 2 dBA for trees)
        veg_reduction = 2
        
        received_db = source_db - distance_reduction - barrier_reduction - veg_reduction
        received_db = max(40, received_db)  # Ambient noise floor
        
        # Compare with standards
        day_limit = 70  # QCVN 26:2010 residential daytime
        night_limit = 55
        
        return {
            "source_db": source_db,
            "received_db": received_db,
            "distance_reduction": distance_reduction,
            "day_compliant": received_db <= day_limit,
            "night_compliant": received_db <= night_limit,
            "day_limit": day_limit,
            "night_limit": night_limit,
        }
    
    def calculate_traffic_impact(
        self,
        daily_trucks: int,
        daily_cars: int,
        road_capacity: int = 1000,  # vehicles per hour
    ) -> Dict[str, Any]:
        """
        Estimate traffic impact.
        
        Args:
            daily_trucks: Number of truck trips per day
            daily_cars: Number of car trips per day
            road_capacity: Road capacity (vehicles/hour)
            
        Returns:
            Dictionary with traffic impact estimates
        """
        # Convert to PCU (Passenger Car Units)
        # Truck = 2.5 PCU, Car = 1 PCU
        total_pcu = daily_trucks * 2.5 + daily_cars * 1.0
        
        # Peak hour factor (assume 15% of daily traffic in peak hour)
        peak_hour_pcu = total_pcu * 0.15
        
        # Level of service
        v_c_ratio = peak_hour_pcu / road_capacity
        
        if v_c_ratio < 0.3:
            los = "A"
            description = "Tự do"
        elif v_c_ratio < 0.5:
            los = "B"
            description = "Ổn định"
        elif v_c_ratio < 0.7:
            los = "C"
            description = "Ổn định"
        elif v_c_ratio < 0.85:
            los = "D"
            description = "Gần bão hòa"
        elif v_c_ratio < 1.0:
            los = "E"
            description = "Bão hòa"
        else:
            los = "F"
            description = "Quá tải"
        
        return {
            "daily_trucks": daily_trucks,
            "daily_cars": daily_cars,
            "total_pcu": total_pcu,
            "peak_hour_pcu": peak_hour_pcu,
            "v_c_ratio": v_c_ratio,
            "level_of_service": los,
            "description": description,
        }
