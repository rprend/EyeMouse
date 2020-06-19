#pragma once

#include "geometry.hpp"

namespace camux {
    enum EyeType {
        Left,
        Right
    };

    class Eye {
    public:
        Eye() :
            type_(Left), coords_{camux::Point(0, 0), camux::Point(0, 0)} {};

        Eye(const EyeType type, camux::Point topLeft, camux::Point bottomRight) : 
            type_(type), coords_{topLeft, bottomRight} {};

        Eye(const EyeType type, const camux::Rectangle coords) : 
            type_(type), coords_(coords) {};


    private:
        EyeType type_;
        camux::Rectangle coords_;
    };
}

