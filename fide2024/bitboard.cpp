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
#include <bitset>
#include <stdint.h>
#include <array>
#include <type_traits>

#include "bitboard.h"
#include "misc.h"


// modified from this code: https://github.com/Gigantua/Chess_Movegen/blob/a76800a55702788ac4f354e6e9fab563b474ec93/Hyperbola.hpp
namespace HyperbolaQsc {
	struct Mask {
		uint64_t diagonal;
		uint64_t antidiagonal;
		uint64_t vertical;
	};

	/* Init */
	std::array<Mask, 64> InitMask() {
		int r = 0, f = 0, i = 0, j = 0, y = 0;
		int d[64]{};

		std::array<Mask, 64> MASK{};

		for (int x = 0; x < 64; ++x) {
			for (y = 0; y < 64; ++y) d[y] = 0;
			// directions
			for (i = -1; i <= 1; ++i)
				for (j = -1; j <= 1; ++j) {
					if (i == 0 && j == 0) continue;
					f = x & 07;
					r = x >> 3;
					for (r += i, f += j; 0 <= r && r < 8 && 0 <= f && f < 8; r += i, f += j) {
						y = 8 * r + f;
						d[y] = 8 * i + j;
					}
				}

			// uint64_t mask
			Mask& mask = MASK[x];
			for (y = x - 9; y >= 0 && d[y] == -9; y -= 9) mask.diagonal |= (1ull << y);
			for (y = x + 9; y < 64 && d[y] == 9; y += 9) mask.diagonal |= (1ull << y);

			for (y = x - 7; y >= 0 && d[y] == -7; y -= 7) mask.antidiagonal |= (1ull << y);
			for (y = x + 7; y < 64 && d[y] == 7; y += 7) mask.antidiagonal |= (1ull << y);

			for (y = x - 8; y >= 0; y -= 8) mask.vertical |= (1ull << y);
			for (y = x + 8; y < 64; y += 8) mask.vertical |= (1ull << y);
		}
		return MASK;
	}

	std::array<uint8_t, 512> InitRank() {

		std::array<uint8_t, 512> rank_attack{};

		for (int x = 0; x < 64; ++x) {
			for (int f = 0; f < 8; ++f) {
				int o = 2 * x;
				int x2{}, y2{};
				int b{};

				y2 = 0;
				for (x2 = f - 1; x2 >= 0; --x2) {
					b = 1 << x2;
					y2 |= b;
					if ((o & b) == b) break;
				}
				for (x2 = f + 1; x2 < 8; ++x2) {
					b = 1 << x2;
					y2 |= b;
					if ((o & b) == b) break;
				}
				rank_attack[x * 8ull + f] = y2;
			}
		}
		return rank_attack;
	}

	std::array<Mask, 64> mask = InitMask();
	std::array<uint8_t, 512> rank_attack = InitRank();

	auto Size = sizeof(mask) + sizeof(rank_attack);

	uint64_t bit_bswap_constexpr(uint64_t b) {
		b = ((b >> 8) & 0x00FF00FF00FF00FFULL) | ((b << 8) & 0xFF00FF00FF00FF00ULL);
		b = ((b >> 16) & 0x0000FFFF0000FFFFULL) | ((b << 16) & 0xFFFF0000FFFF0000ULL);
		b = ((b >> 32) & 0x00000000FFFFFFFFULL) | ((b << 32) & 0xFFFFFFFF00000000ULL);
		return b;
	}

	uint64_t bit_bswap(uint64_t b) {
#if defined(_MSC_VER)
		return _byteswap_uint64(b);
#elif defined(__GNUC__)
		return __builtin_bswap64(b);
#else
		return bit_bswap_constexpr(b);
#endif
	}

	/* Generate attack using the hyperbola quintessence approach */
	uint64_t attack(uint64_t pieces, uint32_t x, uint64_t mask) {
		uint64_t o = pieces & mask;
		return ((o - (1ull << x)) ^ bit_bswap(bit_bswap(o) - (0x8000000000000000ull >> x))) & mask; //Daniel 28.04.2022 - Faster shift. Replaces (1ull << (s ^ 56))
	}

	uint64_t horizontal_attack(uint64_t pieces, uint32_t x) {
		uint32_t file_mask = x & 7;
		uint32_t rank_mask = x & 56;
		uint64_t o = (pieces >> rank_mask) & 126;

		return ((uint64_t)rank_attack[o * 4 + file_mask]) << rank_mask;
	}

	uint64_t vertical_attack(uint64_t occ, uint32_t sq) {
		return attack(occ, sq, mask[sq].vertical);
	}

	uint64_t diagonal_attack(uint64_t occ, uint32_t sq) {
		return attack(occ, sq, mask[sq].diagonal);
	}

	uint64_t antidiagonal_attack(uint64_t occ, uint32_t sq) {
		return attack(occ, sq, mask[sq].antidiagonal);
	}

	uint64_t bishop_attack(int sq, uint64_t occ) {
		return diagonal_attack(occ, sq) | antidiagonal_attack(occ, sq);
	}

	uint64_t rook_attack(int sq, uint64_t occ) {
		return vertical_attack(occ, sq) | horizontal_attack(occ, sq);
	}

	uint64_t Queen(int sq, uint64_t occ) {
		return bishop_attack(sq, occ) | rook_attack(sq, occ);
	}
}


uint8_t PopCnt16[1 << 16];
uint8_t SquareDistance[SQUARE_NB][SQUARE_NB];

Bitboard SquareBB[SQUARE_NB];
Bitboard LineBB[SQUARE_NB][SQUARE_NB];
Bitboard PseudoAttacks[PIECE_TYPE_NB][SQUARE_NB];
Bitboard PawnAttacks[COLOR_NB][SQUARE_NB];


/// Bitboards::pretty() returns an ASCII representation of a bitboard suitable
/// to be printed to standard output. Useful for debugging.

const std::string Bitboards::pretty(Bitboard b) {

  std::string s = "+---+---+---+---+---+---+---+---+\n";

  for (Rank r = RANK_8; r >= RANK_1; --r)
  {
      for (File f = FILE_A; f <= FILE_H; ++f)
          s += b & make_square(f, r) ? "| X " : "|   ";

      s += "|\n+---+---+---+---+---+---+---+---+\n";
  }

  return s;
}


/// Bitboards::init() initializes various bitboard tables. It is called at
/// startup and relies on global objects to be already zero-initialized.

void Bitboards::init() {

  for (unsigned i = 0; i < (1 << 16); ++i)
	  PopCnt16[i] = uint8_t(std::bitset<16>(i).count());

  for (Square s = SQ_A1; s <= SQ_H8; ++s)
      SquareBB[s] = (1ULL << s);

  for (Square s1 = SQ_A1; s1 <= SQ_H8; ++s1)
      for (Square s2 = SQ_A1; s2 <= SQ_H8; ++s2)
          SquareDistance[s1][s2] = std::max(distance<File>(s1, s2), distance<Rank>(s1, s2));

  for (Square s1 = SQ_A1; s1 <= SQ_H8; ++s1)
  {
	  PawnAttacks[WHITE][s1] = pawn_attacks_bb<WHITE>(square_bb(s1));
	  PawnAttacks[BLACK][s1] = pawn_attacks_bb<BLACK>(square_bb(s1));

	  for (int step : {-9, -8, -7, -1, 1, 7, 8, 9})
		  PseudoAttacks[KING][s1] |= safe_destination(s1, step);

	  for (int step : {-17, -15, -10, -6, 6, 10, 15, 17})
		  PseudoAttacks[KNIGHT][s1] |= safe_destination(s1, step);

	  PseudoAttacks[QUEEN][s1] = PseudoAttacks[BISHOP][s1] = attacks_bb<BISHOP>(s1, 0);
	  PseudoAttacks[QUEEN][s1] |= PseudoAttacks[ROOK][s1] = attacks_bb<  ROOK>(s1, 0);

	  for (PieceType pt : { BISHOP, ROOK })
		  for (Square s2 = SQ_A1; s2 <= SQ_H8; ++s2)
			  if (PseudoAttacks[pt][s1] & s2)
				  LineBB[s1][s2] = (attacks_bb(pt, s1, 0) & attacks_bb(pt, s2, 0)) | s1 | s2;
  }
}
