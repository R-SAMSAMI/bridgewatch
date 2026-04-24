from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

FHWA_YEAR = 2025
FHWA_DOWNLOAD_URL = "https://www.fhwa.dot.gov/bridge/nbi/2025hwybronefilenodel.zip"
RAW_ZIP_PATH = RAW_DIR / "fhwa_nbi_2025_all_states.zip"
PROCESSED_DATA_PATH = PROCESSED_DIR / "bridgewatch_2025_processed.csv.gz"

REFERENCE_YEAR = 2025
MODEL_SAMPLE_LIMIT = 120_000

STATE_CODE_TO_NAME = {
    "01": "Alabama",
    "02": "Alaska",
    "04": "Arizona",
    "05": "Arkansas",
    "06": "California",
    "08": "Colorado",
    "09": "Connecticut",
    "10": "Delaware",
    "11": "District of Columbia",
    "12": "Florida",
    "13": "Georgia",
    "15": "Hawaii",
    "16": "Idaho",
    "17": "Illinois",
    "18": "Indiana",
    "19": "Iowa",
    "20": "Kansas",
    "21": "Kentucky",
    "22": "Louisiana",
    "23": "Maine",
    "24": "Maryland",
    "25": "Massachusetts",
    "26": "Michigan",
    "27": "Minnesota",
    "28": "Mississippi",
    "29": "Missouri",
    "30": "Montana",
    "31": "Nebraska",
    "32": "Nevada",
    "33": "New Hampshire",
    "34": "New Jersey",
    "35": "New Mexico",
    "36": "New York",
    "37": "North Carolina",
    "38": "North Dakota",
    "39": "Ohio",
    "40": "Oklahoma",
    "41": "Oregon",
    "42": "Pennsylvania",
    "44": "Rhode Island",
    "45": "South Carolina",
    "46": "South Dakota",
    "47": "Tennessee",
    "48": "Texas",
    "49": "Utah",
    "50": "Vermont",
    "51": "Virginia",
    "53": "Washington",
    "54": "West Virginia",
    "55": "Wisconsin",
    "56": "Wyoming",
    "72": "Puerto Rico",
    "78": "U.S. Virgin Islands",
}

FWF_FIELDS: list[tuple[str, tuple[int, int]]] = [
    ("state_code", (0, 2)),
    ("structure_number", (3, 18)),
    ("year_built", (156, 160)),
    ("lanes_on_structure", (160, 162)),
    ("average_daily_traffic", (164, 170)),
    ("design_load", (174, 175)),
    ("skew", (180, 182)),
    ("type_of_service_on_bridge", (199, 200)),
    ("type_of_service_under_bridge", (200, 201)),
    ("kind_of_material_design", (201, 202)),
    ("type_of_design_construction", (202, 204)),
    ("number_of_spans_in_main_unit", (207, 210)),
    ("length_of_maximum_span", (217, 222)),
    ("structure_length", (222, 228)),
    ("bridge_roadway_width", (234, 238)),
    ("deck_width", (238, 242)),
    ("deck", (258, 259)),
    ("superstructure", (259, 260)),
    ("substructure", (260, 261)),
    ("operating_rating", (264, 267)),
    ("structural_evaluation", (271, 272)),
    ("inspection_date", (286, 290)),
    ("designated_inspection_frequency", (290, 292)),
    ("year_reconstructed", (361, 365)),
    ("average_daily_truck_traffic", (369, 371)),
    ("scour_critical_bridges", (374, 375)),
    ("bridge_condition", (433, 434)),
    ("lowest_condition_code", (434, 435)),
    ("deck_area", (435, 445)),
]
