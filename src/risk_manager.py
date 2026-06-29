"""
Risk management for position sizing and trade validation.
"""

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from config import TradingConfig, MarketConfig
from src.client import Position

if TYPE_CHECKING:
    from src.regime import MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Calculated position size with risk details."""
    size: float
    risk_amount: float
    stop_distance: float
    max_loss: float
    approved: bool
    reason: str


class RiskManager:
    """
    Manages position sizing and risk controls.

    Features:
    - Position sizing based on account risk percentage
    - Maximum position limits
    - Sector exposure limits
    - Daily loss limits
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self.daily_pnl = 0.0
        self.daily_loss_limit = -0.05  # 5% daily loss limit

    def calculate_position_size(
        self,
        account_balance: float,
        stop_distance: float,
        market: MarketConfig,
        regime: Optional["MarketRegime"] = None,
        ig_min_size: Optional[float] = None,
    ) -> PositionSize:
        """
        Calculate appropriate position size based on risk parameters.

        For spread betting:
        - Size = stake per point
        - Risk = size * stop_distance

        Risk budget is divided by max_positions so all slots can
        be filled simultaneously without exceeding available margin.

        Args:
            account_balance: Current account balance in GBP
            stop_distance: Stop loss distance in points
            market: Market configuration
            regime: Optional market regime for size adjustment

        Returns:
            PositionSize with calculated stake
        """
        # Divide risk across max positions so all can coexist
        risk_amount = (account_balance * self.config.risk_per_trade) / self.config.max_positions

        # Calculate size (stake per point)
        # Risk = Size * Stop Distance
        # Size = Risk / Stop Distance
        if stop_distance <= 0:
            return PositionSize(
                size=0.0,
                risk_amount=risk_amount,
                stop_distance=stop_distance,
                max_loss=0.0,
                approved=False,
                reason="Invalid stop distance",
            )

        raw_size = risk_amount / stop_distance

        # Apply regime-based size multiplier
        size_multiplier = 1.0
        regime_info = ""
        if regime:
            from src.regime import get_regime_params
            regime_params = get_regime_params(regime)
            size_multiplier = regime_params.size_multiplier
            regime_info = f" (regime: {regime.code}, multiplier: {size_multiplier})"

            # Block trade entirely if size_multiplier is 0
            if size_multiplier <= 0:
                return PositionSize(
                    size=0.0,
                    risk_amount=risk_amount,
                    stop_distance=stop_distance,
                    max_loss=0.0,
                    approved=False,
                    reason=f"Regime {regime.code} blocks new positions",
                )

        raw_size = raw_size * size_multiplier

        # Minimum dealing size: prefer IG's LIVE minDealSize (threaded in from
        # get_market_info) over the static config default_size. The config value
        # was set 12-25x above IG's true minimum for most indices/FX (e.g. NASDAQ
        # config 0.2 vs IG 0.01, Japan 0.5 vs 0.04), which silently floored valid
        # risk-based sizes up past the £ risk cap and blocked otherwise-tradeable
        # signals (2026-06-29 forensics). Falls back to default_size only when the
        # live snapshot is unavailable. minDealSize also doubles as the size step
        # for spread bets, so snap DOWN to a whole multiple of it — a non-multiple
        # order is rejected by IG, and flooring (not rounding) stays under budget.
        min_size = ig_min_size if (ig_min_size and ig_min_size > 0) else market.default_size
        if min_size and min_size > 0:
            size = round(int(raw_size / min_size) * min_size, 6)
        else:
            size = round(raw_size, 1)

        # Floor to the minimum dealing size if the risk-based figure is smaller.
        # For large-min markets (Copper/Gold, true min 1.0 × a big ATR stop) this
        # can still push the £ loss above budget — the absolute cap below catches
        # that and skips rather than over-risking (2026-06-04 journal forensics).
        floored_up = size < min_size
        if floored_up:
            size = min_size

        # Calculate actual max loss
        max_loss = size * stop_distance

        # Absolute risk-cap: if the min-size floor pushes the £ loss-at-stop above
        # max_risk_gbp, skip rather than over-risk. Only the floor can trip this
        # (risk-based sizing targets the per-trade budget by design, which is set
        # at/under the cap), so a normal-sized trade is never blocked. Absolute £
        # (not a balance×multiple) so the threshold doesn't drift as the account
        # balance moves — see TradingConfig.max_risk_gbp.
        max_allowed_loss = self.config.max_risk_gbp
        if floored_up and max_loss > max_allowed_loss:
            return PositionSize(
                size=0.0,
                risk_amount=risk_amount,
                stop_distance=stop_distance,
                max_loss=max_loss,
                approved=False,
                reason=(
                    f"Min size {min_size} × stop {stop_distance:.1f} risks "
                    f"£{max_loss:.2f} > £{max_allowed_loss:.2f} absolute cap "
                    f"— skipping to avoid over-risk"
                ),
            )

        return PositionSize(
            size=size,
            risk_amount=risk_amount,
            stop_distance=stop_distance,
            max_loss=max_loss,
            approved=True,
            reason=f"Position size calculated{regime_info}",
        )

    def validate_trade(
        self,
        open_positions: list[Position],
        epic: str,
        direction: str,
        account_balance: float,
    ) -> tuple[bool, str]:
        """
        Validate if a new trade should be allowed.

        Checks:
        - Trading enabled
        - Max positions limit
        - Not already in position for this market
        - Daily loss limit not exceeded

        Args:
            open_positions: List of current open positions
            epic: Market EPIC for proposed trade
            direction: Trade direction
            account_balance: Current balance

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check trading enabled
        if not self.config.trading_enabled:
            return False, "Trading is disabled"

        # Check max positions
        if len(open_positions) >= self.config.max_positions:
            return False, f"Max positions ({self.config.max_positions}) reached"

        # Check if already in position for this market
        for pos in open_positions:
            if pos.epic == epic:
                if pos.direction == direction:
                    return False, f"Already {direction} in {epic}"
                # Could allow hedge positions here if desired

        # Check daily loss limit
        daily_loss_pct = self.daily_pnl / account_balance if account_balance > 0 else 0
        if daily_loss_pct <= self.daily_loss_limit:
            return False, f"Daily loss limit reached ({daily_loss_pct:.1%})"

        return True, "Trade validated"

    def update_daily_pnl(self, pnl: float) -> None:
        """Update running daily P&L."""
        self.daily_pnl += pnl
        logger.info(f"Daily P&L updated: {self.daily_pnl:.2f}")

    def reset_daily_pnl(self) -> None:
        """Reset daily P&L (call at start of trading day)."""
        self.daily_pnl = 0.0
        logger.info("Daily P&L reset")

    def get_sector_exposure(
        self,
        positions: list[Position],
        markets: list[MarketConfig],
    ) -> dict[str, float]:
        """
        Calculate exposure by sector.

        Args:
            positions: Open positions
            markets: Market configurations

        Returns:
            Dict of sector -> total exposure
        """
        market_map = {m.epic: m for m in markets}
        exposure: dict[str, float] = {}

        for pos in positions:
            market = market_map.get(pos.epic)
            if market:
                sector = market.sector
                current = exposure.get(sector, 0.0)
                # Exposure = size * current price (approximation)
                exposure[sector] = current + (pos.size * pos.open_level)

        return exposure

    def calculate_portfolio_risk(
        self,
        positions: list[Position],
        account_balance: float,
    ) -> dict:
        """
        Calculate overall portfolio risk metrics.

        Args:
            positions: Open positions
            account_balance: Current balance

        Returns:
            Dict with risk metrics
        """
        total_exposure = 0.0
        total_risk = 0.0
        total_pnl = 0.0

        for pos in positions:
            exposure = pos.size * pos.open_level
            total_exposure += exposure
            total_pnl += pos.profit_loss

            # Risk from stop
            if pos.stop_level:
                stop_distance = abs(pos.open_level - pos.stop_level)
                risk = pos.size * stop_distance
                total_risk += risk

        return {
            "total_exposure": total_exposure,
            "total_risk_at_stop": total_risk,
            "total_pnl": total_pnl,
            "pnl_percentage": (total_pnl / account_balance * 100) if account_balance > 0 else 0,
            "exposure_percentage": (total_exposure / account_balance * 100) if account_balance > 0 else 0,
            "position_count": len(positions),
        }
