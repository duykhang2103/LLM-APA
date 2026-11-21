#include <iostream>
#include <set>
#include <vector>

int main() {
    int length;
    if (!(std::cin >> length)) {
        return 0;
    }

    std::set<int> seen;
    std::vector<int> unique;
    unique.reserve(length);
    for (int i = 0; i < length; ++i) {
        int value;
        std::cin >> value;
        if (seen.insert(value).second) {
            unique.push_back(value);
        }
    }

    std::cout << unique.size() << '\n';
    for (std::size_t i = 0; i < unique.size(); ++i) {
        if (i != 0) {
            std::cout << ' ';
        }
        std::cout << unique[i];
    }
    std::cout << '\n';
}
