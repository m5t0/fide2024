/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2020 The Stockfish developers (see AUTHORS file)

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

//Definition of input features A of NNUE evaluation function

#include "a.h"
#include "index_list.h"

namespace Eval {
    namespace NNUE {
        namespace Features {
            constexpr int NUM_SQ = 64;
            // Orient a square according to perspective (rotates by 180 for black)
            inline Square orient(Color perspective, Square s) {
                return Square(int(s) ^ (bool(perspective) * NUM_SQ));
            }

            // Find the index of the feature quantity from the king position and PieceSquare
            inline IndexType A::MakeIndex(
                Color perspective, Square s, Piece pc) {
                auto piece_index = static_cast<int>((type_of(pc) - 1) * 2 + color_of(pc));
                return IndexType(1 + orient(perspective, s) + piece_index * NUM_SQ);
            }

            // Get a list of indices for active features
            void A::AppendActiveIndices(
                const Position& pos, Color perspective, IndexList* active) {
                Bitboard bb = pos.pieces();
                while (bb) {
                    Square s = pop_lsb(&bb);
                    active->push_back(MakeIndex(perspective, s, pos.piece_on(s)));
                }
            }

            // Get a list of indices for recently changed features
            void A::AppendChangedIndices(
                const Position& pos, Color perspective,
                IndexList* removed, IndexList* added) {

                const auto& dp = pos.state()->dirtyPiece;
                for (int i = 0; i < dp.dirty_num; ++i) {
                    Piece pc = dp.piece[i];
                    if (dp.from[i] != SQ_NONE)
                        removed->push_back(MakeIndex(perspective, dp.from[i], pc));
                    if (dp.to[i] != SQ_NONE)
                        added->push_back(MakeIndex(perspective, dp.to[i], pc));
                }
            }
        }
    }
}  // namespace Eval::NNUE::Features
