#include <iostream>
#include <string>

int main() {
    std::string first, last;
    if (!(std::cin >> first >> last)) {
        return 0;
    }

    std::string best = first.substr(0, 1) + last.substr(0, 1);
    for (std::size_t i = 1; i <= first.size(); ++i) {
        for (std::size_t j = 1; j <= last.size(); ++j) {
            std::string login = first.substr(0, i) + last.substr(0, j);
            if (login < best) {
                best = login;
            }
        }
    }
    std::cout << best << '\n';
}
