#pragma once

#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <functional>
#include <stdexcept>
#include <cstdlib>

// ============================================================================
//     Tiny generic command line parser
// ============================================================================

class ArgParser {
public:
    explicit ArgParser(std::string program_name)
        : program_name_(std::move(program_name))
    {}

    // Register a boolean flag, e.g. --verbose / -v
    void add_flag(
        std::string long_name,
        std::string short_name,
        std::string help,
        bool &target
    ) {
        Option opt;
        opt.long_name = std::move(long_name);
        opt.short_name = std::move(short_name);
        opt.help = std::move(help);
        opt.expects_value = false;
        opt.setter = [&target](std::optional<std::string_view>) {
            target = true;
        };
        options_.push_back(std::move(opt));
    }

    // Register an option that takes a value (string)
    void add_option(
        std::string long_name,
        std::string short_name,
        std::string help,
        std::optional<std::string> &target
    ) {
        Option opt;
        opt.long_name = std::move(long_name);
        opt.short_name = std::move(short_name);
        opt.help = std::move(help);
        opt.expects_value = true;
        opt.setter = [&target](std::optional<std::string_view> val) {
            if (!val) {
                throw std::runtime_error("Missing value for option");
            }
            target = std::string(*val);
        };
        options_.push_back(std::move(opt));
    }

    // Overload for plain std::string target
    void add_option(
        std::string long_name,
        std::string short_name,
        std::string help,
        std::string &target
    ) {
        Option opt;
        opt.long_name = std::move(long_name);
        opt.short_name = std::move(short_name);
        opt.help = std::move(help);
        opt.expects_value = true;
        opt.setter = [&target](std::optional<std::string_view> val) {
            if (!val) {
                throw std::runtime_error("Missing value for option");
            }
            target = std::string(*val);
        };
        options_.push_back(std::move(opt));
    }

    // Overload for int target
    void add_option(
        std::string long_name,
        std::string short_name,
        std::string help,
        int &target
    ) {
        Option opt;
        opt.long_name = std::move(long_name);
        opt.short_name = std::move(short_name);
        opt.help = std::move(help);
        opt.expects_value = true;
        opt.setter = [&target](std::optional<std::string_view> val) {
            if (!val) {
                throw std::runtime_error("Missing value for option");
            }
            std::string s(val->begin(), val->end());
            try {
                target = std::stoi(s);
            } catch (...) {
                throw std::runtime_error("Invalid integer for option");
            }
        };
        options_.push_back(std::move(opt));
    }

    // Parse argv and fill options without positional arguments
    void parse(
        int argc,
        char **argv
    ) const {
        std::vector<std::string> positional;
        parse(argc, argv, positional);
    }

    // Parse argv and fill options + positional arguments
    void parse(
        int argc,
        char **argv,
        std::vector<std::string> &positional
    ) const {
        for (int i = 1; i < argc; ++i) {
            std::string_view arg = argv[i];

            // Help first
            if (arg == "--help" || arg == "-h") {
                print_help();
                std::exit(0);
            }

            if (arg.size() > 2 && arg.rfind("--", 0) == 0) {
                // Long option: --name or --name=value
                auto eq_pos = arg.find('=');
                std::string_view name = arg.substr(0, eq_pos);
                std::optional<std::string_view> value;

                if (eq_pos != std::string_view::npos) {
                    value = arg.substr(eq_pos + 1);
                }

                Option const* opt = find_long(name);
                if (!opt) {
                    throw std::runtime_error("Unknown option: " + std::string(name));
                }

                if (opt->expects_value && !value) {
                    if (i + 1 >= argc) {
                        throw std::runtime_error("Missing value for option: " + std::string(name));
                    }
                    value = std::string_view(argv[++i]);
                } else if (!opt->expects_value && value) {
                    throw std::runtime_error("Option does not take a value: " + std::string(name));
                }

                opt->setter(value);
            } else if (arg.size() > 1 && arg[0] == '-' && arg[1] != '-') {
                // Short option: -v or -t 4
                std::string name = std::string(arg.substr(0, 2)); // just -x style, no bundling
                Option const* opt = find_short(name);
                if (!opt) {
                    throw std::runtime_error("Unknown option: " + name);
                }

                std::optional<std::string_view> value;
                if (opt->expects_value) {
                    if (arg.size() > 2) {
                        // Handle -t8 style? For simplicity, treat as error.
                        throw std::runtime_error("Use '-t 8' or '--threads=8', not '-t8'");
                    }
                    if (i + 1 >= argc) {
                        throw std::runtime_error("Missing value for option: " + name);
                    }
                    value = std::string_view(argv[++i]);
                }

                opt->setter(value);
            } else {
                // Positional argument
                positional.emplace_back(arg);
            }
        }
    }

    void print_help() const {
        std::cout << "Usage: " << program_name_ << " [options] [positional args]\n\n";
        std::cout << "Options:\n";
        std::cout << "  -h, --help\n      Show this help message\n";
        for (auto const& opt : options_) {
            std::string names = "  ";
            if (!opt.short_name.empty()) {
                names += opt.short_name + ", ";
            } else {
                names += "    ";
            }
            names += opt.long_name;
            if (opt.expects_value) {
                names += " VALUE";
            }

            std::cout << names << "\n";
            if (!opt.help.empty()) {
                std::cout << "      " << opt.help << "\n";
            }
        }
        std::cout << "\n";
    }

private:
    struct Option {
        std::string long_name;
        std::string short_name;
        std::string help;
        bool expects_value = false;
        std::function<void(std::optional<std::string_view>)> setter;
    };

    Option const* find_long(std::string_view name) const {
        for (auto const& opt : options_) {
            if (opt.long_name == name) {
                return &opt;
            }
        }
        return nullptr;
    }

    Option const* find_short(std::string_view name) const {
        for (auto const& opt : options_) {
            if (!opt.short_name.empty() && opt.short_name == name) {
                return &opt;
            }
        }
        return nullptr;
    }

    std::string program_name_;
    std::vector<Option> options_;
};
