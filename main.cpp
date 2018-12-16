#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>

class Node;

class Classifier
{
    public:
        std::vector<Node *> nodes;
        unsigned int size;

        ~Classifier();
        std::string classify(std::string);
        double calculate(unsigned int *);
};

class Node
{
    friend class Classifier;
    public:
        Node(double, std::vector<double> &);
    private:
        double key;
        std::vector<double> features;
};

Classifier::~Classifier()
{
    while(!nodes.empty())
    {
        delete nodes.at(nodes.size() - 1);

        nodes.pop_back();
    }
}

std::string Classifier::classify(std::string input)
{
    std::cout << "This dataset has " << size << " features with " << nodes.size() << " instances\n" << std::endl;
    std::cout << "Running nearest neighbor classifier with K = 1 cross evaluation\n" << std::endl;
    std::cout << "Beginning ";
    if(input == "1" || input == "2")
    {
        unsigned int option;

        if(input == "1")
        {
            option = 0;
            std::cout << "Forward Selection";
        }
        else
        {
            option = 1;
            std::cout << "Backward Elimination";
        }
        std::cout << " Search\n" << std::endl;
        std::cout << "--------------------------------------------------\n" << std::endl;

        unsigned int * features = new unsigned int[size] {};
        unsigned int * owner = new unsigned int[size] {};

        for(unsigned int i = 0; i < size; i++)
        {
            *(features + i) = option;
            *(owner + i) = option;
        }

        double max = 0.0;

        for(unsigned int i = 0; i < size; i++)
        {
            std::string print = "";

            if(input == "2")
            {
                if(i == 0)
                {
                    for(unsigned int j = 0; j < size; j++)
                    {
                        if(!print.empty())
                        {
                            print += ", ";
                        }
                        print += std::to_string(j + 1);
                    }
                    std::cout << "Using feature(s) {" << print << "}, accuracy is ";
                    max = calculate(features);
                    std::cout << std::setprecision(4) << max * 100.0 << '%' << '\n' << std::endl;
                    std::cout << "Feature set {" << print << "} was the best subset with an accuracy of " << std::setprecision(4) << max * 100.0 << '%' << '\n' << std::endl;
                    std::cout << "--------------------------------------------------\n" << std::endl;
                }
                else if(i == size - 1)
                {
                    break;
                }
            }

            unsigned int index;
            double local = 0.0;

            for(unsigned int j = 0; j < size; j++)
            {
                double percent = 0.0;

                if(features[j] == option)
                {
                    features[j] = (option & ~1) | (~option & 1);
                    print.clear();
                    for(unsigned int k = 0; k < size; k++)
                    {
                        if(features[k] == 1)
                        {
                            if(!print.empty())
                            {
                                print += ", ";
                            }
                            print += std::to_string(k + 1);
                        }
                    }
                    std::cout << "Using feature(s) {" << print << "}, accuracy is ";
                    percent = calculate(features);
                    std::cout << std::setprecision(4) << percent * 100.0 << '%' << std::endl;
                    features[j] = option;
                }
                if(percent >= local)
                {
                    index = j;
                    local = percent;
                    if(local >= max)
                    {
                        for(unsigned int k = 0; k < size; k++)
                        {
                            *(owner + k) = features[k];
                        }
                        *(owner + j) = (option & ~1) | (~option & 1);
                        max = local;
                    }
                }
            }
            features[index] = (option & ~1) | (~option & 1);

            unsigned int j;

            for(j = 0; j < size; j++)
            {
                if(features[j] != owner[j])
                {
                    break;
                }
            }
            std::cout << std::endl;
            if(j != size)
            {
                std::cout << "[Warning] Accuracy has decreased. Continuing search in case of local maxima." << std::endl;
            }
            print.clear();
            for(unsigned int j = 0; j < size; j++)
            {
                if(features[j] == 1)
                {
                    if(!print.empty())
                    {
                        print += ", ";
                    }
                    print += std::to_string(j + 1);
                }
            }
            std::cout << "Feature set {" << print << "} was the best subset with an accuracy of " << std::setprecision(4) << local * 100.0 << '%' << '\n' << std::endl;
            std::cout << "--------------------------------------------------\n" << std::endl;
        }

        std::string print = "";

        for(unsigned int i = 0; i < size; i++)
        {
            if(owner[i] == 1)
            {
                if(!print.empty())
                {
                    print += ", ";
                }
                print += std::to_string(i + 1);
            }
        }
        std::cout << "[Results] The best feature set is {" << print << "} with an accuracy of " << std::setprecision(4) << max * 100.0 << '%' << '\n' << std::endl;
    }
    do
    {
        std::cout << "----- [Input Option] -----" << std::endl;
        std::cout << "[1] Select new search algorithm" << std::endl;
        std::cout << "[2] Select new data set" << std::endl;
        std::cout << "[3] Exit program" << std::endl;
        std::cout << "> ";
        std::cin >> input;
        std::cout << std::endl;
        if(input != "1" && input != "2" && input != "3")
        {
            std::cout << "[Error] Invalid Syntax\n" << std::endl;
            input.clear();
        }
    }while(input.empty());

    return input;
}

double Classifier::calculate(unsigned int * features)
{
    unsigned int correct = 0;

    for(unsigned int i = 0; i < nodes.size(); i++)
    {
        int index = -1;
        double min;

        for(unsigned int j = 0; j < nodes.size(); j++)
        {
            if(j != i)
            {
                double distance = 0.0;

                for(unsigned int l = 0; l < size; l++)
                {
                    if(features[l] == 1)
                    {
                        distance += pow(nodes.at(i)->features.at(l) - nodes.at(j)->features.at(l), 2.0);
                    }
                }
                if(sqrt(distance) < min || index == -1)
                {
                    index = j;
                    min = sqrt(distance);
                }
            }
        }
        if(nodes.at(index)->key == nodes.at(i)->key)
        {
            correct += 1;
        }
    }

    return double(correct) / double(nodes.size());
}

Node::Node(double key, std::vector<double> & features)
{
    this->key = key;
    this->features = features;
}

int main()
{
    std::cout << "----- [Welcome to the Feature Selection Program] -----\n" << std::endl;

    std::string input;

    do
    {
        std::ifstream read;

        do
        {
            std::cout << "Input data set file name:" << std::endl;
            std::cout << "> ";
            std::cin >> input;
            std::cout << std::endl;
            input = "inputs/" + input;
            read.open(input);
            if(!read)
            {
                std::cout << "[Error] Failed to read " << input << " or " << input << " does not exist\n" << std::endl;
            }
        }while(!read);

        Classifier * classifier = new Classifier();

        for(input; getline(read, input); )
        {
            std::istringstream iss(input);
            std::vector<double> data;
            double number;

            while(iss >> number)
            {
                data.push_back(number);
            }
            
            double key = data.front();

            data.erase(data.begin());
            classifier->size = data.size();

            classifier->nodes.push_back(new Node(key, data));
        }
        read.close();
        do
        {
            std::cout << "----- [Enter Feature Selection Algorithm] -----" << std::endl;
            std::cout << "[1] Forward Selection" << std::endl;
            std::cout << "[2] Backward Elimination" << std::endl;
            std::cout << "[3] Custom" << std::endl;
            std::cout << "> ";
            std::cin >> input;
            std::cout << std::endl;
            if(input == "1" || input == "2" || input == "3")
            {
                input = classifier->classify(input);
            }
            else
            {
                std::cout << "[Error] Invalid Syntax\n" << std::endl;
                input.clear();
            }
            if(input == "1")
            {
                input.clear();
            }
        }while(input.empty());
    }while(input != "3");
    std::cout << "----- [Terminating Program] -----" << std::endl;

    return 0;
}