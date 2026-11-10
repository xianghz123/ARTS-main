"""ARTS: aggregate-guided road-network trajectory synthesis (Porto E1/E2)."""
from .data import PortoData, load_porto
from .synthesis import synthesize
from .metrics import evaluate_e1, evaluate_e2
