//
// Created by wrk on 2024/1/11.
//
#include <iostream>
#include <Eigen/Core>
#include "../include/utility.hpp"

using Scalar = float;

#define Q_LAMBDA //是否构建Q_labmda来执行回退

void small_test() {
    Eigen::Matrix<Scalar, 3, 2> A;
    A << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;

    std::cout << "Original Matrix A:\n" << A << "\n\n";
    // 构造 Givens 旋转矩阵
    Eigen::JacobiRotation<Scalar> gr;

    gr.makeGivens(A(0,0), A(2,0));
    A.applyOnTheLeft(0, 2, gr.adjoint());
    std::cout << "Matrix A after G1:\n" << A << "\n";
    A(2, 0) = 0;

    gr.makeGivens(A(0,0), A(1,0));
    A.applyOnTheLeft(0, 1, gr.adjoint());
    std::cout << "Matrix A after G2:\n" << A << "\n";
    A(1, 0) = 0;

    gr.makeGivens(A(1,1), A(2,1));
    A.applyOnTheLeft(1, 2, gr.adjoint());
    std::cout << "Matrix A after G3:\n" << A << "\n";
    A(2, 1) = 0;
    std::cout << "Matrix A after Givens rotation:\n" << A << "\n";
}

void big_test() {
    Eigen::Matrix<Scalar, 9, 16> A;
    A << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0, 0, 0, 0, 0, 0, 37, 38, 39, 46,
            7, 8, 9, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 40, 41, 47,
            13, 14, 15, 16, 17, 18, 0, 0, 0, 0, 0, 0, 0, 0, 42, 48,
            0, 0, 0, 0, 0, 0, 19, 20, 21, 22, 23, 24, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 25, 26, 27, 28, 29, 30,0,0,0,0,
            0, 0, 0, 0, 0, 0, 31, 32, 33, 34, 35, 36,0,0,0,0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 43,0,0,0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,44,0,0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,45,0;

    std::cout << "Original Matrix A:\n" << A << "\n\n";
    // 构造 Givens 旋转矩阵
    Eigen::JacobiRotation<Scalar> gr;
#ifndef Q_LAMBDA
    Scalar G1_arr;
    gr.makeGivens(A(0,12), A(6,12), &G1_arr);
    Eigen::JacobiRotation<Scalar> G1 = gr.adjoint();
    A.applyOnTheLeft(0,6, G1);
    std::cout << "Matrix A after G1:\n" << A << "\n";
    A(6, 12) = 0;

    std::cout << "G1_arr: " << G1_arr << std::endl;
    std::cout << "Original Rotation cos: " << gr.c() << "\t" << " sin:\t" << gr.s() << "\n\n";
    std::cout << "Givens Adjoint cos: " << gr.adjoint().c() << "\t" << " sin:\t" << gr.adjoint().s() << "\n\n";
    std::cout << "Givens transpose cos: " << gr.transpose().c() << "\t" << " sin:\t" << gr.transpose().s() << "\n\n";
    std::cout << "Givens Rotation:\n" << gr.adjoint().c() << "\t" << -gr.adjoint().s() << "\n\n"
              << gr.adjoint().s() << "\t" << gr.adjoint().c() << "\n\n";
    //针对实数，adjoint()和transpose()是等价的，makeGivens()根据传入参数cos和sin，我们可以外部构建出Givens，相当于把向量逆时针旋转：
    // cos  -sin
    // cos   sin
    //applyOnTheLeft()中不知道为什么只能用.adjoint()，.adjoint()实际上构建了一个新的对象，其中cos取共轭，sin取反，JacobiRotation(conj(m_c), -m_s)
    //只不过在applyOnTheLeft()中可能有根据A的行数构建一个主对角线为1，其他为0的真正可左乘到A上的Givens矩阵
    //至于joan sola的公式(5.16)，可能和Eigen中的实现有些出入，其cos的符号对不上公式，后面再说

    gr.makeGivens(A(1,13), A(7,13));
    Eigen::JacobiRotation<Scalar> G2 = gr.adjoint();
    A.applyOnTheLeft(1,7, G2);
    std::cout << "Matrix A after G2:\n" << A << "\n\n";
//    A(7, 13) = 0;

    gr.makeGivens(A(1,13), A(6,13));
    Eigen::JacobiRotation<Scalar> G3 = gr.adjoint();
    A.applyOnTheLeft(1,6, G3);
    std::cout << "Matrix A after G3:\n" << A << "\n\n";
//    A(6, 13) = 0;

    gr.makeGivens(A(2,14), A(8,14));
    Eigen::JacobiRotation<Scalar> G4 = gr.adjoint();
    A.applyOnTheLeft(2,8, G4);
    std::cout << "Matrix A after G4:\n" << A << "\n\n";
//    A(8, 14) = 0;

    gr.makeGivens(A(2,14), A(7,14));
    Eigen::JacobiRotation<Scalar> G5 = gr.adjoint();
    A.applyOnTheLeft(2,7, G5);
    std::cout << "Matrix A after G5:\n" << A << "\n\n";
//    A(7, 14) = 0;

    gr.makeGivens(A(2,14), A(6,14));
    Eigen::JacobiRotation<Scalar> G6 = gr.adjoint();
    A.applyOnTheLeft(2,6, G6);
    std::cout << "Matrix A after G6:\n" << A << "\n\n";
//    A(6, 14) = 0;
#else
    Eigen::Matrix<Scalar, 9, 16> B = A; //B用于跟使用Q_lambda作对比
    Eigen::Matrix<Scalar, 9, 16> C = A;
    Eigen::Matrix<Scalar, 9, 9> Q_lambda = Eigen::Matrix<Scalar, 9, 9>::Identity();//整体Q_lambda
    Eigen::Matrix<Scalar, 9, 9> Q_lambda_C = Eigen::Matrix<Scalar, 9, 9>::Identity();//整体Q_lambda
    //G1
    gr.makeGivens(A(0,12), A(6,12));
    Eigen::Matrix<Scalar, 9, 9> tmpQ = Eigen::Matrix<Scalar, 9, 9>::Identity();
    tmpQ(0,0) = gr.c();
    tmpQ(0,6) = -gr.s();
    tmpQ(6,0) = gr.s();
    tmpQ(6,6) = gr.c();
    Q_lambda = tmpQ * Q_lambda;
    A = tmpQ * A;
    C.applyOnTheLeft(0,6, gr.adjoint());
    Q_lambda_C.applyOnTheLeft(0,6, gr.adjoint());
    std::cout << "G1 tmpQ: \n" << tmpQ << "\n\n";
    std::cout << "after G1 Q_lambda: \n" << Q_lambda << "\n\n";
    std::cout << "Matrix A after G1:\n" << A << "\n\n";

    //G2
    gr.makeGivens(A(1,13), A(7,13));
    tmpQ.setIdentity();
    tmpQ(1,1) = gr.c();
    tmpQ(1,7) = -gr.s();
    tmpQ(7,1) = gr.s();
    tmpQ(7,7) = gr.c();
    Q_lambda = tmpQ * Q_lambda;
    A = tmpQ * A;
    C.applyOnTheLeft(1,7, gr.adjoint());
    Q_lambda_C.applyOnTheLeft(1,7, gr.adjoint());
    std::cout << "G2 tmpQ: \n" << tmpQ << "\n\n";
    std::cout << "after G2 Q_lambda: \n" << Q_lambda << "\n\n";
    std::cout << "Matrix A after G2:\n" << A << "\n\n";

    //G3
    gr.makeGivens(A(1,13), A(6,13));
    tmpQ.setIdentity();
    tmpQ(1,1) = gr.c();
    tmpQ(1,6) = -gr.s();
    tmpQ(6,1) = gr.s();
    tmpQ(6,6) = gr.c();
    Q_lambda = tmpQ * Q_lambda;
    A = tmpQ * A;
    C.applyOnTheLeft(1,6, gr.adjoint());
    Q_lambda_C.applyOnTheLeft(1,6, gr.adjoint());
    std::cout << "G3 tmpQ: \n" << tmpQ << "\n\n";
    std::cout << "after G3 Q_lambda: \n" << Q_lambda << "\n\n";
    std::cout << "Matrix A after G3:\n" << A << "\n\n";

    //G4
    gr.makeGivens(A(2,14), A(8,14));
    tmpQ.setIdentity();
    tmpQ(2,2) = gr.c();
    tmpQ(2,8) = -gr.s();
    tmpQ(8,2) = gr.s();
    tmpQ(8,8) = gr.c();
    Q_lambda = tmpQ * Q_lambda;
    A = tmpQ * A;
    C.applyOnTheLeft(2,8, gr.adjoint());
    Q_lambda_C.applyOnTheLeft(2,8, gr.adjoint());
    std::cout << "G4 tmpQ: \n" << tmpQ << "\n\n";
    std::cout << "after G4 Q_lambda: \n" << Q_lambda << "\n\n";
    std::cout << "Matrix A after G4:\n" << A << "\n\n";

    //G5
    gr.makeGivens(A(2,14), A(7,14));
    tmpQ.setIdentity();
    tmpQ(2,2) = gr.c();
    tmpQ(2,7) = -gr.s();
    tmpQ(7,2) = gr.s();
    tmpQ(7,7) = gr.c();
    Q_lambda = tmpQ * Q_lambda;
    A = tmpQ * A;
    C.applyOnTheLeft(2,7, gr.adjoint());
    Q_lambda_C.applyOnTheLeft(2,7, gr.adjoint());
    std::cout << "G5 tmpQ: \n" << tmpQ << "\n\n";
    std::cout << "after G5 Q_lambda: \n" << Q_lambda << "\n\n";
    std::cout << "Matrix A after G5:\n" << A << "\n\n";

    //G6
    gr.makeGivens(A(2,14), A(6,14));
    tmpQ.setIdentity();
    tmpQ(2,2) = gr.c();
    tmpQ(2,6) = -gr.s();
    tmpQ(6,2) = gr.s();
    tmpQ(6,6) = gr.c();
    Q_lambda = tmpQ * Q_lambda;
    A = tmpQ * A;
    C.applyOnTheLeft(2,6, gr.adjoint());
    Q_lambda_C.applyOnTheLeft(2,6, gr.adjoint());
    std::cout << "G6 tmpQ: \n" << tmpQ << "\n\n";
    std::cout << "after G6 Q_lambda: \n" << Q_lambda << "\n\n";
    std::cout << "Matrix A after G6:\n" << A << "\n\n";

    //执行旋转
    std::cout << "Matrix A==B before Givens rotation:\n" << B << "\n\n";
    B = Q_lambda * B;
    std::cout << "Matrix A after Givens rotation:\n" << A << "\n\n";
    std::cout << "Matrix B after Givens rotation:\n" << B << "\n\n";
    std::cout << "Matrix C after Givens rotation:\n" << C << "\n\n";
#endif


#ifndef Q_LAMBDA
    //使用6个Givens矩阵rollback G1.T * G2.T * G3.T * G4.T * G5.T * G6.T * A
    A.applyOnTheLeft(2,6, G6.transpose());
    A.applyOnTheLeft(2,7, G5.transpose());
    A.applyOnTheLeft(2,8, G4.transpose());
    A.applyOnTheLeft(1,6, G3.transpose());
    A.applyOnTheLeft(1,7, G2.transpose());
    A.applyOnTheLeft(0,6, G1.transpose());
    std::cout << "Matrix A after Givens.T rotation rollback:\n" << A << "\n\n";
#else
    //使用Q_lambda.T 执行rollback
    A = Q_lambda.transpose() * A;
    std::cout << "Matrix A after Givens.T rotation rollback:\n" << A << "\n\n";

    B = Q_lambda.transpose() * B;
    std::cout << "Matrix B after Givens.T rotation rollback:\n" << B << "\n\n";

    C = Q_lambda_C.transpose() * C;
    std::cout << "Matrix C after Givens.T rotation rollback:\n" << C << "\n\n";

#endif
}

int main(int argc, char** argv) {
//    small_test();   //小规模测试
    big_test();     //大规模测试
}