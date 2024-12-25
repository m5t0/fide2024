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

#ifndef UCI_H_INCLUDED
#define UCI_H_INCLUDED

#include <map>
#include <string>

#include "types.h"

std::vector<std::string> white_split(const std::string& str);

class StringSplitter {
public:
    StringSplitter(const std::string& s) :buffer(white_split(s)), idx(0) {}
    bool empty() const {
        return idx == buffer.size();
    }
    std::string next_token() {
        return buffer[idx++];
    }

private:
    std::vector<std::string> buffer;
    size_t idx;
};

class Position;

namespace OptionValue {

    // min: -100, max: 100
    constexpr int Contempt = 24;
    constexpr int Threads = 1;
    // 64bitÇ»ÇÁç≈ëÂ131072ÅA32bitÇ»ÇÁç≈ëÂ2048
    constexpr int Hash = 16;
    constexpr bool Ponder = true;
    constexpr int MultiPV = 1;
    // min: 0, max: 5000
    constexpr int MoveOverhead = 30;
    // min: 0, max: 5000
    constexpr int MinimumThinkingTime = 20;
    // min: 10, max: 1000
    constexpr int SlowMover = 84;
}

namespace UCI {
void loop(int argc, char* argv[]);
std::string square(Square s);
std::string move(Move m);
Move to_move(const Position& pos, std::string& str);

#ifndef KAGGLE
std::string value(Value v);
std::string pv(const Position& pos, Depth depth, Value alpha, Value beta);

#endif // KAGGLE

} // namespace UCI

#endif // #ifndef UCI_H_INCLUDED
