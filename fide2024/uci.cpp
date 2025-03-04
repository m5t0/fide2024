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

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <cctype>

#ifndef KAGGLE

#include <cmath>

#endif // !KAGGLE


#include "evaluate.h"
#include "movegen.h"
#include "position.h"
#include "search.h"
#include "thread.h"
#include "timeman.h"
#include "tt.h"
#include "uci.h"

using namespace std;

namespace UCI {
    Move best_move = MOVE_NULL;
    Move ponder_move = MOVE_NULL;
    bool output_best_move = true;
    std::vector<std::string> str;
    std::string allocated_time = "", fen = "", last_move = "";
    bool is_bench = false;
}

#ifndef KAGGLE

extern vector<string> setup_bench(const Position&, istream&);
#endif // !KAGGLE

namespace {

    // FEN string of the initial position, normal chess
    const char* StartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";


    // position() is called when engine receives the "position" UCI command.
    // The function sets up the position described in the given FEN string ("fen")
    // or the starting position ("startpos") and then makes the moves given in the
    // following move list ("moves").

    void position(Position& pos, istringstream& is, StateListPtr& states) {

        Move m;
        string token, fen;

        is >> token;

        if (token == "startpos")
        {
            fen = StartFEN;
            is >> token; // Consume "moves" token if any
        }
        else if (token == "fen")
            while (is >> token && token != "moves")
                fen += token + " ";
        else
            return;

        states = StateListPtr(new std::deque<StateInfo>(1)); // Drop old and create a new one
        pos.set(fen, &states->back());

        // Parse move list (if any)
        while (is >> token && (m = UCI::to_move(pos, token)) != MOVE_NONE)
        {
            states->emplace_back();
            pos.do_move(m, states->back());
        }
    }

    // go() is called when engine receives the "go" UCI command. The function sets
    // the thinking time and other parameters from the input string, then starts
    // the search.

    void go(Position& pos, istringstream& is, StateListPtr& states) {

        Search::LimitsType limits;
        string token;
        bool ponderMode = false;

        limits.startTime = now(); // As early as possible!

        while (is >> token)
            if (token == "searchmoves") // Needs to be the last command on the line
                while (is >> token)
                    limits.searchmoves.push_back(UCI::to_move(pos, token));

            else if (token == "wtime")     is >> limits.time[WHITE];
            else if (token == "btime")     is >> limits.time[BLACK];
            else if (token == "winc")      is >> limits.inc[WHITE];
            else if (token == "binc")      is >> limits.inc[BLACK];
            else if (token == "movestogo") is >> limits.movestogo;
            else if (token == "depth")     is >> limits.depth;
            else if (token == "nodes")     is >> limits.nodes;
            else if (token == "movetime")  is >> limits.movetime;
            else if (token == "mate")      is >> limits.mate;
            else if (token == "perft")     is >> limits.perft;
            else if (token == "infinite")  limits.infinite = 1;
            else if (token == "ponder")    ponderMode = true;

        Threads.start_thinking(pos, states, limits, ponderMode);

#ifdef KAGGLE
        if(!ponderMode)
            Threads.main()->wait_for_search_finished();

        if (!UCI::is_bench && !ponderMode && UCI::ponder_move != MOVE_NULL) {
            auto states = StateListPtr(new std::deque<StateInfo>(1));
            pos.set(UCI::fen, &states->back());
            StateInfo si2;
            pos.do_move(UCI::best_move, si2);
            StateInfo si3;
            pos.do_move(UCI::ponder_move, si3);

            //sync_cout << "best_move:" << UCI::move(UCI::best_move) << " ponder_move:" << UCI::move(UCI::ponder_move) << " fen:" << pos.fen() << sync_endl;
            UCI::str.emplace_back("position fen " + pos.fen() + "\n");
            UCI::str.emplace_back("go ponder wtime " + UCI::allocated_time + " btime " + UCI::allocated_time + "\n");
        }
#endif // KAGGLE
    }


    // bench() is called when engine receives the "bench" command. Firstly
    // a list of UCI commands is setup according to bench parameters, then
    // it is run one by one printing a summary at the end.

#ifndef KAGGLE
    void bench(Position& pos, istream& args, StateListPtr& states) {
        string token;
        uint64_t num, nodes = 0, cnt = 1;

        vector<string> list = setup_bench(pos, args);
        num = count_if(list.begin(), list.end(), [](string s) { return s.find("go ") == 0 || s.find("eval") == 0; });

        TimePoint elapsed = now();

        for (const auto& cmd : list)
        {
            istringstream is(cmd);
            is >> skipws >> token;

            if (token == "go" || token == "eval")
            {
                cerr << "\nPosition: " << cnt++ << '/' << num << endl;
                if (token == "go")
                {
                    go(pos, is, states);
                    Threads.main()->wait_for_search_finished();
                    nodes += Threads.nodes_searched();
                }
                else
                    sync_cout << "\n" << Eval::trace(pos) << sync_endl;
            }
            else if (token == "position")   position(pos, is, states);
            else if (token == "ucinewgame") { Search::clear(); elapsed = now(); } // Search::clear() may take some while
        }

        elapsed = now() - elapsed + 1; // Ensure positivity to avoid a 'divide by zero'

        dbg_print(); // Just before exiting

        cerr << "\n==========================="
            << "\nTotal time (ms) : " << elapsed
            << "\nNodes searched  : " << nodes
            << "\nNodes/second    : " << 1000 * nodes / elapsed << endl;
    }

    // The win rate model returns the probability (per mille) of winning given an eval
    // and a game-ply. The model fits rather accurately the LTC fishtest statistics.
    int win_rate_model(Value v, int ply) {

        // The model captures only up to 240 plies, so limit input (and rescale)
        double m = std::min(240, ply) / 64.0;

        // Coefficients of a 3rd order polynomial fit based on fishtest data
        // for two parameters needed to transform eval to the argument of a
        // logistic function.
        double as[] = { -8.24404295, 64.23892342, -95.73056462, 153.86478679 };
        double bs[] = { -3.37154371, 28.44489198, -56.67657741,  72.05858751 };
        double a = (((as[0] * m + as[1]) * m + as[2]) * m) + as[3];
        double b = (((bs[0] * m + bs[1]) * m + bs[2]) * m) + bs[3];

        // Transform eval to centipawns with limited range
        double x = Utility::clamp(double(100 * v) / PawnValueEg, -1000.0, 1000.0);

        // Return win rate in per mille (rounded to nearest)
        return int(0.5 + 1000 / (1 + std::exp((a - x) / b)));
    }

#endif

} // namespace


/// UCI::loop() waits for a command from stdin, parses it and calls the appropriate
/// function. Also intercepts EOF from stdin to ensure gracefully exiting if the
/// GUI dies unexpectedly. When called with some command line arguments, e.g. to
/// run 'bench', once the command is executed the function returns immediately.
/// In addition to the UCI ones, also some additional debug commands are supported.

void UCI::loop(int argc, char* argv[]) {

    Position pos;
    string token, cmd;
    StateListPtr states(new std::deque<StateInfo>(1));

    pos.set(StartFEN, &states->back());

    //for (int i = 1; i < argc; ++i)
    //    cmd += std::string(argv[i]) + " ";

    is_bench = argc > 1;

    do {
        if (argc == 1 && !getline(cin, cmd)) // Block here waiting for input or EOF
            cmd = "quit";

        istringstream is(cmd);

#ifdef KAGGLE
        if (!is_bench) {
            str.clear();
            is >> skipws >> allocated_time >> fen;
            std::string t;
            while (t != "last_move") {
                is >> skipws >> t;
                if(t != "last_move")
                    fen += " " + t;
            }
            is >> last_move;

            if (UCI::ponder_move == MOVE_NULL) {
                str.emplace_back("position fen " + fen + "\n");
                //str.emplace_back("isready\n");
                str.emplace_back("go wtime " + allocated_time + " btime " + allocated_time + "\n");
            }
            else {
                if (last_move == UCI::move(UCI::ponder_move)) {
                    str.emplace_back("ponderhit\n");
                }
                else {
                    str.emplace_back("stop\n");
                    str.emplace_back("position fen " + fen + "\n");
                    str.emplace_back("go wtime " + allocated_time + " btime " + allocated_time + "\n");
                }
            }
        }
        else {
#endif // KAGGLE
            str.clear();
            str.emplace_back(cmd);
#ifdef KAGGLE
        }
#endif // KAGGLE

        for (size_t i = 0; i < str.size(); i++) {
            auto& val = str[i];
            is = istringstream(val);

            token.clear(); // Avoid a stale if getline() returns empty or blank line
            is >> skipws >> token;

            if (token == "quit" || token == "stop") {
#ifdef KAGGLE
                if (token == "stop") {
                    UCI::output_best_move = false;
                }
#endif // KAGGLE
                    
                Threads.stop = true;
                Threads.main()->wait_for_search_finished();

#ifdef KAGGLE
                if (token == "stop") {
                    UCI::output_best_move = true;
                }
#endif // KAGGLE
            }

            // The GUI sends 'ponderhit' to tell us the user has played the expected move.
            // So 'ponderhit' will be sent if we were told to ponder on the same move the
            // user has played. We should continue searching but switch from pondering to
            // normal search.
            else if (token == "ponderhit") {
                Threads.main()->ponder = false; // Switch to normal search
#ifdef KAGGLE
                Threads.main()->wait_for_search_finished();

                if (UCI::ponder_move != MOVE_NULL) {
                    StateInfo si2;
                    pos.do_move(UCI::best_move, si2);
                    StateInfo si3;
                    pos.do_move(UCI::ponder_move, si3);

                    //sync_cout << "best_move:" << UCI::move(UCI::best_move) << " ponder_move:" << UCI::move(UCI::ponder_move) << " fen:" << pos.fen() << sync_endl;
                    UCI::str.emplace_back("position fen " + pos.fen() + "\n");
                    UCI::str.emplace_back("go ponder wtime " + UCI::allocated_time + " btime " + UCI::allocated_time + "\n");
                }
#endif // KAGGLE
            }

            else if (token == "uci")
                sync_cout
#ifndef KAGGLE
                << "id name " << engine_info(true)
#endif // !KAGGLE
                << "\nuciok" << sync_endl;

            else if (token == "go")         go(pos, is, states);
            else if (token == "position")   position(pos, is, states);
            else if (token == "ucinewgame") Search::clear();
            else if (token == "isready")    sync_cout << "readyok" << sync_endl;

            // Additional custom non-UCI commands, mainly for debugging.
            // Do not use these commands during a search!
#ifndef KAGGLE
            else if (token == "flip")     pos.flip();
            else if (token == "bench")    bench(pos, is, states);
            else if (token == "d")        sync_cout << pos << sync_endl;
            else if (token == "eval")     sync_cout << Eval::trace(pos) << sync_endl;
            else if (token == "compiler") sync_cout << compiler_info() << sync_endl;
#endif // !KAGGLE
            else
                sync_cout << "Unknown command: " << cmd << sync_endl;
        }

    } while (token != "quit" && argc == 1); // Command line args are one-shot
}


/// UCI::value() converts a Value to a string suitable for use with the UCI
/// protocol specification:
///
/// cp <x>    The score from the engine's point of view in centipawns.
/// mate <y>  Mate in y moves, not plies. If the engine is getting mated
///           use negative values for y.

#ifndef KAGGLE
string UCI::value(Value v) {

    assert(-VALUE_INFINITE < v && v < VALUE_INFINITE);

    stringstream ss;

    if (abs(v) < VALUE_MATE_IN_MAX_PLY)
        ss << "cp " << v * 100 / PawnValueEg;
    else
        ss << "mate " << (v > 0 ? VALUE_MATE - v + 1 : -VALUE_MATE - v) / 2;

    return ss.str();
}

/// UCI::wdl() report WDL statistics given an evaluation and a game ply, based on
/// data gathered for fishtest LTC games.

string UCI::wdl(Value v, int ply) {

    stringstream ss;

    int wdl_w = win_rate_model(v, ply);
    int wdl_l = win_rate_model(-v, ply);
    int wdl_d = 1000 - wdl_w - wdl_l;
    ss << " wdl " << wdl_w << " " << wdl_d << " " << wdl_l;

    return ss.str();
}

#endif // !KAGGLE


/// UCI::square() converts a Square to a string in algebraic notation (g1, a7, etc.)

std::string UCI::square(Square s) {
    return std::string{ char('a' + file_of(s)), char('1' + rank_of(s)) };
}


/// UCI::move() converts a Move to a string in coordinate notation (g1f3, a7a8q).
/// The only special case is castling, where we print in the e1g1 notation in
/// normal chess mode, and in e1h1 notation in chess960 mode. Internally all
/// castling moves are always encoded as 'king captures rook'.

string UCI::move(Move m) {

    Square from = from_sq(m);
    Square to = to_sq(m);

    if (m == MOVE_NONE)
        return "(none)";

    if (m == MOVE_NULL)
        return "0000";

    if (type_of(m) == CASTLING)
        to = make_square(to > from ? FILE_G : FILE_C, rank_of(from));

    string move = UCI::square(from) + UCI::square(to);

    if (type_of(m) == PROMOTION)
        move += " pnbrqk"[promotion_type(m)];

    return move;
}


/// UCI::to_move() converts a string representing a move in coordinate notation
/// (g1f3, a7a8q) to the corresponding legal Move, if any.

Move UCI::to_move(const Position& pos, string& str) {

    if (str.length() == 5) // Junior could send promotion piece in uppercase
        str[4] = char(tolower(str[4]));

    for (const auto& m : MoveList<LEGAL>(pos))
        if (str == UCI::move(m))
            return m;

    return MOVE_NONE;
}
