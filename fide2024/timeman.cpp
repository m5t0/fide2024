/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2008 Tord Romstad (Glaurung author)
  Copyright (C) 2008-2015 Marco Costalba, Joona Kiiski, Tord Romstad
  Copyright (C) 2015-2020 Marco Costalba, Joona Kiiski, Gary Linscott, Tord Romstad

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <algorithm>
#include <cfloat>
#include <cmath>

#include "search.h"
#include "timeman.h"
#include "uci.h"

TimeManagement Time; // Our global time management object

namespace {

  enum TimeType { OptimumTime, MaxTime };

  constexpr int MoveHorizon   = 50;   // Plan time management at most this many moves ahead
  constexpr double MaxRatio   = 7.3;  // When in trouble, we can step over reserved time with this ratio
  constexpr double StealRatio = 0.34; // However we must not steal time from remaining moves over this ratio


  // move_importance() is a skew-logistic function based on naive statistical
  // analysis of "how many games are still undecided after n half-moves". Game
  // is considered "undecided" as long as neither side has >275cp advantage.
  // Data was extracted from the CCRL game database with some simple filtering criteria.

  double move_importance(int ply) {

    constexpr double XScale = 6.85;
    constexpr double XShift = 64.5;
    constexpr double Skew   = 0.171;

    return pow((1 + exp((ply - XShift) / XScale)), -Skew) + DBL_MIN; // Ensure non-zero
  }

  template<TimeType T>
  TimePoint remaining(TimePoint myTime, int movesToGo, int ply, TimePoint slowMover) {

    constexpr double TMaxRatio   = (T == OptimumTime ? 1.0 : MaxRatio);
    constexpr double TStealRatio = (T == OptimumTime ? 0.0 : StealRatio);

    double moveImportance = (move_importance(ply) * slowMover) / 100.0;
    double otherMovesImportance = 0.0;

    for (int i = 1; i < movesToGo; ++i)
        otherMovesImportance += move_importance(ply + 2 * i);

    double ratio1 = (TMaxRatio * moveImportance) / (TMaxRatio * moveImportance + otherMovesImportance);
    double ratio2 = (moveImportance + TStealRatio * otherMovesImportance) / (moveImportance + otherMovesImportance);

    return TimePoint(myTime * std::min(ratio1, ratio2)); // Intel C++ asks for an explicit cast
  }

} // namespace


/// TimeManagement::init() is called at the beginning of the search and calculates
/// the bounds of time allowed for the current game ply. We currently support:
//      1) x basetime (+ z increment)
//      2) x moves in y seconds (+ z increment)

void TimeManagement::init(Search::LimitsType& limits, Color us, int ply) {

  TimePoint moveOverhead = OptionValue::MoveOverhead;
  TimePoint slowMover = OptionValue::SlowMover;

  // opt_scale is a percentage of available time to use for the current move.
  // max_scale is a multiplier applied to optimumTime.
  double opt_scale, max_scale;

  startTime = limits.startTime;

  // Maximum move horizon of 50 moves
  int mtg = limits.movestogo ? std::min(limits.movestogo, 50) : 50;

  // Make sure timeLeft is > 0 since we may use it as a divisor
  TimePoint timeLeft =  std::max(TimePoint(1),
      limits.time[us] + limits.inc[us] * (mtg - 1) - moveOverhead * (2 + mtg));

  // A user may scale time usage by setting UCI option "Slow Mover"
  // Default is 100 and changing this value will probably lose elo.
  timeLeft = slowMover * timeLeft / 100;

  // x basetime (+ z increment)
  // If there is a healthy increment, timeLeft can exceed actual available
  // game time for the current move, so also cap to 20% of available game time.
  if (limits.movestogo == 0)
  {
      opt_scale = std::min(0.008 + std::pow(ply + 3.0, 0.5) / 250.0,
          0.2 * limits.time[us] / double(timeLeft));
      max_scale = std::min(7.0, 4.0 + ply / 12.0);
  }
  // x moves in y seconds (+ z increment)
  else
  {
      opt_scale = std::min((0.8 + ply / 128.0) / mtg,
          0.8 * limits.time[us] / double(timeLeft));
      max_scale = std::min(6.3, 1.5 + 0.11 * mtg);
  }

  // Never use more than 80% of the available time for this move
  optimumTime = TimePoint(opt_scale * timeLeft);
  maximumTime = TimePoint(std::min(0.8 * limits.time[us] - moveOverhead, max_scale * optimumTime));

  if (OptionValue::Ponder)
      optimumTime += optimumTime / 4;
}
