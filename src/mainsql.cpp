#include <sqlite3.h>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>

#include "qTensor.hpp"
#include "Contraction.hpp"

#ifdef ENABLE_CUDA
#include "qTensorCUDA.cuh"
#endif

using namespace std;

bool retrieveData(sqlite3* db, std::vector<Contraction>& contractions, int programId) {
    sqlite3_stmt* stmt;
    int rc;

    std::string query = "\
        SELECT c.id, c.program_id, c.span, c.left_id, c.right_id, c.kind, g.data  \
        FROM contractions as c left join gates as g on c.gate_id == g.id  \
        WHERE program_id = ?";
    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Error preparing statement: " << sqlite3_errmsg(db) << std::endl;
        return false;
    }

    sqlite3_bind_int(stmt, 1, programId);

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        Contraction contraction;
        contraction.id = sqlite3_column_int(stmt, 0);
        contraction.programId = sqlite3_column_int(stmt, 1);
        string span_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        span_str = span_str.substr(1, span_str.size() - 2);
        size_t pos = 0;

        // TODO : ------------------------------------------------------ fix implicitly tensor expanded gates
        auto span = std::vector<unsigned char>();
        while ((pos = span_str.find(',')) != string::npos) {
            span.push_back(stoi(span_str.substr(0, pos)));
            span_str.erase(0, pos + 1);
        }
        span.push_back(stoi(span_str));
        auto start = span[0];
        auto end = span[span.size() - 1];
        for (unsigned char i = start; i <= end; i++) contraction.span.push_back(i);
        // TODO : ------------------------------------------------------ fix implicitly tensor expanded gates

        contraction.leftId = sqlite3_column_int(stmt, 3);
        contraction.rightId = sqlite3_column_int(stmt, 4);
        contraction.kind = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        if (sqlite3_column_type(stmt, 6) != SQLITE_NULL) {
            const void* data = sqlite3_column_blob(stmt, 6);
            size_t dataSize = sqlite3_column_bytes(stmt, 6);
            std::vector<double> values(static_cast<const double*>(data), static_cast<const double*>(data) + dataSize / sizeof(double));
            contraction.data = QTensor(values, contraction.span);
        }
        contractions.push_back(contraction);
    }

    sqlite3_finalize(stmt);

    for (auto& contraction : contractions) {
        if (contraction.kind == "C") {
            for (auto& c : contractions) {
                if (c.id == contraction.leftId) {
                    contraction.left = &c;
                }
                if (c.id == contraction.rightId) {
                    contraction.right = &c;
                }
            }
        }
    }

    if (rc != SQLITE_DONE) {
        std::cerr << "Error retrieving contractions: " << sqlite3_errmsg(db) << std::endl;
        return false;
    }
    
    if (rc != SQLITE_DONE) {
        std::cerr << "Error retrieving gates: " << sqlite3_errmsg(db) << std::endl;
        return false;
    }

    return true;
}

ostream& operator<<(ostream& os, const vector<unsigned char>& span) {
    os << "[";
    for (size_t i = 0; i < span.size(); ++i) {
        os << (size_t)span[i];
        if (i < span.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

void printTree(Contraction* root, size_t level = 0) {
    if (root == nullptr) {
        return;
    }

    std::cout << std::string(level * 2, ' ') << "Contraction: " << root->span << " (Kind: " << root->kind << ")" << std::endl;

    if (root->kind == "G") {
        std::cout << std::string((level + 1) * 2, ' ') << "Gate " << " rank: " << (root->data).getRank() << ")" << std::endl;
    } else {
        printTree(root->left, level + 1);
        printTree(root->right, level + 1);
    }
} 

void contractTree(Contraction* root) {
    // cout << "Contracting tree node with id " << root->id << " and kind " << root->kind << endl;
    if (root == nullptr)
        return;
    if (root->kind == "C") {
        // cout << "Contracting left child with pointer " << root->left << " and right child with pointer " << root->right << endl;
        contractTree(root->left);
        contractTree(root->right);
        // cout << "Getting data for contraction " << root->id << " with left data rank " << root->left->data.getRank() << " and right data rank " << root->right->data.getRank() << endl;

        // root->data = QTensor::contraction(root->right->data, root->left->data);
        // root->data = QTensor::contraction(root->left->data, root->right->data);
        // root->data = contractionGPU(root->left->data, root->right->data);

        #ifdef ENABLE_CUDA
        //root->data = contractionGPU(root->right->data, root->left->data);
        #else
        root->data = QTensor::contraction(root->right->data, root->left->data);
        #endif
    }
}

#define DEBUG false

int main(int argc, char** argv) {
    sqlite3* db;
    int rc = sqlite3_open("../data/db_ours.sqlite", &db);
    if (rc != SQLITE_OK) {
        std::cerr << "Error opening database: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return 1;
    }

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <program_id>" << std::endl;
        sqlite3_close(db);
        return 1;
    }

    #ifdef ENABLE_CUDA
    cout << "CUDA enabled" << endl;
    #else
    cout << "CUDA disabled" << endl;
    #endif

    std::vector<Contraction> contractions;
    // MOD
    int programId = std::stoi(argv[1]);

    if (retrieveData(db, contractions, programId)) {
        if (DEBUG){
        for (const auto& contraction : contractions) {
            std::cout << "Contraction: " << contraction.id << std::endl;
            std::cout << "Span: " << contraction.span << std::endl;
            std::cout << "Left ID: " << contraction.leftId << std::endl;
            std::cout << "Right ID: " << contraction.rightId << std::endl;
            std::cout << "Kind: " << contraction.kind << std::endl;
            std::cout << "Data rank: " << contraction.data.getRank() << std::endl;
            std::cout << "Data:\n";
            contraction.data.printValues();
            std::cout << std::endl;
        }
        }
    }

    cout << "-----------------------------------" << endl;
    cout << "Starting contraction" << endl;


    // root is last

    Contraction* root = contractions.empty() ? nullptr : &contractions.back();
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    #ifdef ENABLE_CUDA
    contractTreeGPU(root);
    #else
    contractTree(root);
    #endif
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout << "Time difference = " << (double)chrono::duration_cast<chrono::microseconds>(end - begin).count() / 1000. << "[ms]" << endl;
    // append the time and the program id to times.csv
    ofstream times("times.csv", ios::app);
    times << (double)chrono::duration_cast<chrono::microseconds>(end - begin).count() / 1000. << "," << programId << endl;
    times.close();
    if (DEBUG){
        for (const auto& contraction : contractions) {
            std::cout << "Contraction: " << contraction.id << std::endl;
            std::cout << "Span: " << contraction.span << std::endl;
            std::cout << "Left ID: " << contraction.leftId << std::endl;
            std::cout << "Right ID: " << contraction.rightId << std::endl;
            std::cout << "Kind: " << contraction.kind << std::endl;
            std::cout << "Data rank: " << contraction.data.getRank() << std::endl;
            std::cout << "Data:\n"; 
            contraction.data.printValues();
            std::cout << std::endl;
        }
    }

    ofstream out("unitary.txt");
    (contractions.end()-1)->data.printValues(out);
    out.close();

    sqlite3_close(db);
    return 0;
}