#ifndef CHECKPOINT_HPP
#define CHECKPOINT_HPP

#include <cstdint>
#include <filesystem>
#include <string>
#include <simplemc/mc/json.hpp>
#include <simplemc/mc/simulation_stats.hpp>
#include <simplemc/serialize/concepts.hpp>
#include "diagram/diagram_config.hpp"
#include "diagram/diagram_io.hpp"

// Checkpoint/restart of diagramc: one file per rank, <prefix>_checkpoints/rank<r>.msgpack (MessagePack: compact, and
// NaN/Inf survive, unlike in text JSON), written atomically by simplemc (temporary file + rename). Content, through
// simplemc::save_json_checkpoint:
//     "rng"           the chain's xoshiro256** state
//     "stats"         production steps and time so far (simulation_stats)
//     "updates"       per update: weight, ratio, proposal/acceptance counters, SignCounter (updates/dmc_update_set.hpp)
//     "measurements"  per estimator: is_active, accumulator and counters (the estimators' simplemc_save hooks)
//     "config"        the diagram (DiagramState: lines in slot order, segment momenta)
//     "dmc"           Meta below
// A restart is statistical: the diagram's cached quantities are rebuilt from its lines and momenta, so the continued
// chain matches an uninterrupted one up to rounding. The thermalization phase is not saved beyond its outcome: a job
// killed during thermalization starts again from scratch.
namespace checkpoint {

    struct Meta {
        int format {1};
        int n_ranks {0};
        int rank {0};
        std::string phase {"thermalized"};             // "thermalized" (before production) or "production"
        bool thermalization_enabled {false};
        bool thermalized {false};
        bool therm_timed_out {false};
        std::uint64_t therm_steps {0};
        double therm_time {0.};
        std::uint64_t n_negative {0};                 // cfg.n_negative: production steps on negative diagrams
        int jobs {1};                                  // jobs that have contributed to this chain, this one included
        std::string fingerprint;                       // input::fingerprint of the input the state belongs to
    };

    template <simplemc::serializer S>
    void simplemc_save(S s, const Meta & m){
        s.save_at("format", m.format);
        s.save_at("n_ranks", m.n_ranks);
        s.save_at("rank", m.rank);
        s.save_at("phase", m.phase);
        s.save_at("thermalization_enabled", m.thermalization_enabled);
        s.save_at("thermalized", m.thermalized);
        s.save_at("therm_timed_out", m.therm_timed_out);
        s.save_at("therm_steps", m.therm_steps);
        s.save_at("therm_time", m.therm_time);
        s.save_at("n_negative", m.n_negative);
        s.save_at("jobs", m.jobs);
        s.save_at("fingerprint", m.fingerprint);
    }

    template <simplemc::serializer S>
    void simplemc_load(const S & s, Meta & m){
        s.load_at("format", m.format);
        s.load_at("n_ranks", m.n_ranks);
        s.load_at("rank", m.rank);
        s.load_at("phase", m.phase);
        s.load_at("thermalization_enabled", m.thermalization_enabled);
        s.load_at("thermalized", m.thermalized);
        s.load_at("therm_timed_out", m.therm_timed_out);
        s.load_at("therm_steps", m.therm_steps);
        s.load_at("therm_time", m.therm_time);
        s.load_at("n_negative", m.n_negative);
        s.load_at("jobs", m.jobs);
        s.load_at("fingerprint", m.fingerprint);
    }

    inline const simplemc::json_io_options io {.mode = simplemc::json_file_mode::msgpack, .indent = 0};

    // <prefix>_checkpoints/: one directory per run, so the rank files do not crowd the output directory
    inline std::filesystem::path directory(const std::string & prefix){
        return prefix + "_checkpoints";
    }

    inline std::filesystem::path path(const std::string & prefix, int rank){
        return directory(prefix) / ("rank" + std::to_string(rank) + ".msgpack");
    }

    // the layout before the subdirectory, <prefix>_checkpoint_rank<r>.msgpack: still read by a restart
    inline std::filesystem::path legacyPath(const std::string & prefix, int rank){
        return prefix + "_checkpoint_rank" + std::to_string(rank) + ".msgpack";
    }

    // The metadata and the diagram alone: what a restart needs before the estimators can be built.
    inline void readHeader(const std::filesystem::path & file, Meta & meta, DiagramState & diagram){
        simplemc::load_json_file(file, [&](const simplemc::json_serializer & s){
            s.load_at("dmc", meta);
            s.load_at("config", diagram);
        }, io);
    }

    template <typename RNG, typename US, typename MS>
    void save(const std::filesystem::path & file, const RNG & rng, const US & us, const MS & ms, const simplemc::simulation_stats & stats,
              const diagram_cfg & cfg, const Meta & meta){
        const DiagramState diagram {captureDiagram(cfg)};
        if (file.has_parent_path()) { std::filesystem::create_directories(file.parent_path()); }    // every rank, no ordering needed
        simplemc::save_json_checkpoint(file, rng, us, ms, stats, diagram,
                                       [&](simplemc::json_serializer & s){ s.save_at("dmc", meta); }, io);
    }

    // Everything else (the diagram is restored separately, from readHeader, before the estimators exist).
    template <typename RNG, typename US, typename MS>
    void load(const std::filesystem::path & file, RNG & rng, US & us, MS & ms, simplemc::simulation_stats & stats){
        DiagramState unused;
        simplemc::load_json_checkpoint(file, rng, us, ms, stats, unused, [](const simplemc::json_serializer &){}, io);
    }
}

#endif // !CHECKPOINT_HPP
